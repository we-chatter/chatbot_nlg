#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:cch
# datetime:2020/11/24 下午4:31

import torch
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F


class DialogueGenerator:
    def __init__(self,
                 model_path,
                 device,
                 vocab_path,
                 max_history_len=5,
                 max_len=300,
                 repetition_penalty=1.0,
                 temperature=1,
                 topk=40,
                 topp=0.31,
                 log_path=None):

        if log_path is not None:
            self.logger = self.create_logger(log_path)
        self.tokenizer = self.get_tokenizer(vocab_path=vocab_path)
        self.model = self.get_model(model_path).to(device)
        self.max_history_len = max_history_len
        self.device = device
        self.max_len = max_len
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.topk = topk
        self.topp = topp

    @classmethod
    def get_model(cls, model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        return model

    @classmethod
    def get_tokenizer(cls, vocab_path):
        return BertTokenizer(vocab_file=vocab_path)

    @classmethod
    def create_logger(cls, log_path):
        """
        将日志输出到日志文件和控制台
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        # 创建一个handler，用于写入日志文件
        file_handler = logging.FileHandler(
            filename=log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # 创建一个handler，用于将日志输出到控制台
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logger.addHandler(console)
        return logger

    def top_k_top_p_filtering(self, logits, filter_value=-float('Inf')):
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(self.topk, logits.size(-1))  # Safety check
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

        if self.topp > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.topp
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def chat(self, text: str, history: list):
        history.append(self.tokenizer.encode(text))
        input_ids = [self.tokenizer.cls_token_id]
        for history_id, history_utr in enumerate(history[-self.max_history_len:]):
            input_ids.extend(history_utr)
            input_ids.append(self.tokenizer.sep_token_id)
        current_input = torch.tensor(input_ids).to(device=self.device, dtype=torch.long)
        generate_list = []
        for _ in range(self.max_len):
            outputs = self.model(current_input)
            next_token_logits = outputs[0][-1, :]
            for id in set(generate_list):
                next_token_logits[id] /= self.repetition_penalty
            next_token_logits = next_token_logits / self.temperature
            next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = self.top_k_top_p_filtering(next_token_logits)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == self.tokenizer.sep_token_id:
                break
            generate_list.append(next_token.item())
            current_input = torch.cat((current_input, next_token), dim=0)

        history.append(generate_list)
        text2 = self.tokenizer.convert_ids_to_tokens(generate_list)
        text2 = "".join(text2)
        return text2, history


if __name__ == '__main__':
    model_path = "/home/cch/PycharmProjects/nlg_chatbot/dialogue_model"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_path = "vocabulary/vocab_small.txt"
    dialogue_generator = DialogueGenerator(model_path, device, vocab_path)
    history = []
    while True:
        text = input("user input text:")
        if text == "q":
            break
        answer, history = dialogue_generator.chat(text=text, history=history)
        print("chat bot answer: ", "".join(answer))
