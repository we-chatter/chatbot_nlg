#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:cch
# datetime:2020/11/24 上午11:19
import torch

import random
import numpy as np
import argparse

from chatbot.models.trainer import Trainer

PAD = '[PAD]'
pad_id = 0
logger = None

parser = argparse.ArgumentParser()

parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                    help='选择模型参数')
parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
parser.add_argument('--raw_process_path', default='data/train_input.txt', type=str, required=False, help='原始训练语料')

parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='训练日志存放位置')
parser.add_argument('--epochs', default=10, type=int, required=False, help='训练的轮次')
parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                    help='对话模型输出路径')
parser.add_argument('--pretrained_model', default='./dialogue_model', type=str, required=False, help='预训练的GPT2模型的路径')
parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")

args = parser.parse_args()


def set_random_seed(seed):
    """
    设置训练的随机种子
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    if args.seed:
        set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        log_path=args.log_path,
        vocab_path=args.vocab_path,
        raw_process_path=args.raw_process_path,
        pre_trained_model=args.pretrained_model,
        device=device
    )
    trainer.train()


if __name__ == '__main__':
    main()
