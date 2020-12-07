#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:cch
# datetime:2020/11/24 上午10:15
from abc import ABC

from torch.utils.data import Dataset
import torch


class ChatBotDataSet(Dataset, ABC):

    def __init__(self, file_path, n_ctx=300):
        super(ChatBotDataSet, self).__init__()
        with open(file_path, 'r') as f:
            self.data_list = f.readlines()
        self.n_ctx = n_ctx

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        dialogue_ids = self.data_list[idx].strip().split(" ")
        dialogue_ids = [int(item) for item in dialogue_ids]
        dialogue_ids = dialogue_ids[:self.n_ctx]
        return dialogue_ids
