#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:cch
# datetime:2020/11/24 上午11:19
import transformers
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from chatbot.dataset import ChatBotDataSet

PAD = '[PAD]'
pad_id = 0


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


class Trainer:
    def __init__(self,
                 log_path,
                 vocab_path,
                 raw_process_path,
                 device=None,
                 model_config=None,
                 pre_trained_model=None,
                 batch_size=8,
                 lr=1e-4,
                 epochs=50,
                 gradient_accumulation=1,
                 writer_dir="./summary",
                 dialogue_model_output_path="./model",
                 max_grad_norm=1.0,
                 warmup_steps=2000
                 ):
        self.warmup_steps = warmup_steps
        self.logger = self.create_logger(log_path)
        self.tokenizer = self.get_tokenizer(vocab_path)
        self.vocab_size = len(self.tokenizer)
        self.model, self.n_ctx = self.create_model(pre_trained_model, self.vocab_size, model_config)
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model, device_ids=[0, 1, 2, 3, 4, 5, 6])

        train_data_set = ChatBotDataSet(file_path=raw_process_path, n_ctx=self.n_ctx)
        self.total_steps = int(train_data_set.__len__() * epochs / batch_size / gradient_accumulation)

        self.train_data_loader = DataLoader(train_data_set,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=20,
                                            collate_fn=collate_fn)

        self.optimizer = transformers.AdamW(self.model.parameters(), lr=lr, correct_bias=True)
        self.scheduler = transformers.WarmupLinearSchedule(self.optimizer, warmup_steps=self.warmup_steps,
                                                           t_total=self.total_steps)
        self.device = device
        self.tb_writer = SummaryWriter(writer_dir)
        self.epochs = epochs
        self.gradient_accumulation = gradient_accumulation
        self.dialogue_model_output_path = dialogue_model_output_path
        self.max_grad_norm = max_grad_norm

    @classmethod
    def get_tokenizer(cls, vocab_path):
        tokenizer = BertTokenizer(vocab_path)
        return tokenizer

    @classmethod
    def create_logger(cls, log_path):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(
            filename=log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logger.addHandler(console)

        return logger

    def create_model(self, pre_trained_model, vocab_size, model_config=None):

        if pre_trained_model:  # if the pretrained model is give
            model = GPT2LMHeadModel.from_pretrained(pre_trained_model)
        else:  # if there is no pretrained model we will give the model config
            model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(model_config)
            model = GPT2LMHeadModel(config=model_config)
        # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
        model.resize_token_embeddings(vocab_size)
        self.logger.info('model config:\n{}'.format(model.config.to_json_string()))
        return model, model.config.to_dict().get("n_ctx")

    def calculate_loss_and_accuracy(self, outputs, labels):
        logits = outputs[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(self.device)

        loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy

    def train(self):
        self.model.to(self.device)
        self.model.train()
        # 计算所有epoch进行参数优化的总步数total_steps
        self.logger.info('total training steps = {}'.format(self.total_steps))

        self.logger.info('starting training')
        # 用于统计每次梯度累计的loss
        running_loss = 0
        # 统计一共训练了多少个step
        overall_step = 0
        # 记录tensorboardX
        # tb_writer = SummaryWriter(log_dir=args.writer_dir)
        # 记录 out of memory的次数
        oom_time = 0
        # 开始训练
        for epoch in range(self.epochs):
            epoch_start_time = datetime.now()
            for batch_idx, input_ids in enumerate(self.train_data_loader):
                # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
                # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
                input_ids = input_ids.to(self.device)
                # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
                try:
                    outputs = self.model.forward(input_ids=input_ids)
                    loss, accuracy = self.calculate_loss_and_accuracy(outputs, labels=input_ids)

                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()
                        accuracy = accuracy.mean()
                    if self.gradient_accumulation > 1:
                        loss = loss / self.gradient_accumulation
                        accuracy = accuracy / self.gradient_accumulation
                    loss.backward()
                    # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    # 进行一定step的梯度累计之后，更新参数
                    if (batch_idx + 1) % self.gradient_accumulation == 0:
                        running_loss += loss.item()
                        # 更新参数
                        self.optimizer.step()
                        # 清空梯度信息
                        self.optimizer.zero_grad()
                        # 进行warm up
                        self.scheduler.step()
                        overall_step += 1
                        # 更新日志与tnesorboardX信息
                        if (overall_step + 1) % 100 == 0:
                            self.logger.info(
                                "batch {} of epoch {}, loss {}, accuracy {}".format(batch_idx + 1, epoch + 1, loss,
                                                                                    accuracy))
                            self.tb_writer.add_scalar('loss', loss.item(), overall_step)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        self.logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        self.logger.info(str(exception))
                        raise exception
            self.logger.info('saving model for epoch {}'.format(epoch + 1))

            model_path = join(self.dialogue_model_output_path, 'model_epoch{}.bin'.format(epoch + 1))
            if not os.path.exists(self.dialogue_model_output_path):
                os.mkdir(self.dialogue_model_output_path)
            if (epoch + 1) % 5 == 0:
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_save.save_pretrained(model_path)
            self.logger.info('epoch {} finished'.format(epoch + 1))
            epoch_finish_time = datetime.now()
            self.logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
        self.logger.info('training finished')

    def evaluate(self, device, test_file, multi_gpu, args):
        self.logger.info("start evaluating model")
        self.model.eval()
        self.logger.info('starting evaluating')
        test_data_set = ChatBotDataSet(file_path=test_file,  n_ctx=self.n_ctx)

        test_data_loader = DataLoader(test_data_set,
                                      batch_size=12,
                                      shuffle=True,
                                      num_workers=20,
                                      collate_fn=collate_fn)
        with torch.no_grad():
            for batch_idx, input_ids in enumerate(test_data_loader):
                input_ids.to(device)
                outputs = self.model.forward(input_ids=input_ids)
                loss, accuracy = self.calculate_loss_and_accuracy(outputs, labels=input_ids)

                if multi_gpu:
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                    accuracy = accuracy / args.gradient_accumulation
                self.logger.info("evaluate batch {} ,loss {} ,accuracy {}".format(batch_idx, loss, accuracy))
                # tb_writer.add_scalar('loss', loss.item(), overall_step)
            self.logger.info("finishing evaluating")
