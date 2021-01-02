#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:cch

import argparse
import datetime
from sanic import Sanic
from sanic.response import json
from chatbot.generate.generate_dialogue import DialogueGenerator
import torch

app = Sanic("chatting platform")

cache = {}

parser = argparse.ArgumentParser()
parser.add_argument("-mp",
                    "--model_path",
                    dest="model_path",
                    help="The GPT2 model path",
                    required=True,
                    type=str,
                    # default='/home/cch/PycharmProjects/nlg_chatbot/dialogue_model'
                    )

parser.add_argument("-pmp",
                    "--poetry_model_path",
                    dest="poetry_model_path",
                    help="The GPT2 poetry model path",
                    required=True,
                    type=str
                    )

parser.add_argument("-p",
                    "--port",
                    dest="port",
                    help="The server port",
                    required=True,
                    type=int)

parser.add_argument("-uc",
                    "--use_cuda",
                    dest="use_cuda",
                    help="use cuda in the process",
                    type=bool
                    )

parser.add_argument("-vp",
                    "--vocab_path",
                    dest="vocab_path",
                    help="give the vacab path",
                    required=True,
                    type=str,
                    # default="/home/cch/PycharmProjects/nlg_chatbot/vocabulary/vocab_small.txt"
                    )

parser.add_argument("-nh",
                    "--n_history",
                    dest="n_history",
                    help="give the history chatting",
                    required=True,
                    type=int,
                    # default="/home/cch/PycharmProjects/nlg_chatbot/vocabulary/vocab_small.txt"
                    )

args = parser.parse_args()


@app.route("/generate/chat")
async def gen_txt(request):
    generator = DialogueGenerator(model_path=args.model_path,
                                  device=torch.device(
                                      "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"),
                                  vocab_path=args.vocab_path
                                  )
    req = request.json
    # history = req["history"]
    user_id = req["history"]
    text = req["text"]
    history = cache[user_id] if user_id in cache else []
    gen_text, gen_history = generator.chat(text=text, history=history)
    cache[user_id] = gen_history if len(gen_history) <= args.n_history else gen_history[-args.n_history:]
    msg = f"{text} from {user_id} and quest is {gen_text}, and the date is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    generator.write_log(msg)
    return json({
        "text": gen_text
    })


@app.route("/generate/poetry")
async def gen_poetry(request):
    generator = DialogueGenerator(model_path=args.poetry_model_path,
                                  device=torch.device(
                                      "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"),
                                  vocab_path=args.vocab_path
                                  )
    req = request.json
    # history = req["history"]
    text = req["text"]
    gen_text, gen_history = generator.chat(text=text, history=None)
    return json({
        "text": gen_text
    })


@app.route("/generate/couplet")
async def gen_poetry(request):
    generator = DialogueGenerator(model_path=args.couplet_model_path,
                                  device=torch.device(
                                      "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"),
                                  vocab_path=args.vocab_path
                                  )
    req = request.json
    # history = req["history"]
    text = req["text"]
    gen_text, gen_history = generator.chat(text=text, history=None)
    return json({
        "text": gen_text
    })


def main():
    app.run(host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
