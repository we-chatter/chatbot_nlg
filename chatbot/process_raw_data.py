#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:cch
# datetime:2020/11/26 上午11:24

import fcntl
import threading
import time

from tqdm import tqdm
from transformers import BertTokenizer

mutex = threading.Lock()


def write_2_txt(f, text, tokenizer, line_signal, length, index):
    with mutex:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        utterances = text.split(line_signal)
        dialogue_ids = [tokenizer.cls_token_id]
        for utterance in utterances:
            dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
            dialogue_ids.append(tokenizer.sep_token_id)

        # dialogue_ids = dialogue_ids[:n_ctx]
        for dialogue_id in dialogue_ids:
            f.write(str(dialogue_id) + ' ')
        if index < length - 1:
            f.write("\n")
            # time.sleep(0.1)


def transformer(raw_path, txt_path, vocab_path):
    with open(raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
        line_signal = "\r\n"
    else:
        train_data = data.split("\n\n")
        line_signal = "\n"
    tokenizer = BertTokenizer(vocab_path)
    n_ctx = 300
    with open(txt_path, "a+") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            write_thread = threading.Thread(
                target=write_2_txt,
                args=(f, dialogue, tokenizer, line_signal, len(train_data), dialogue_index,)
                )
            write_thread.start()


if __name__ == '__main__':
    raw_path = "./data/train.txt"
    txt_path = "./data/train_input.txt"
    vocab_path = "./vocabulary/vocab_small.txt"
    transformer(raw_path, txt_path, vocab_path)


