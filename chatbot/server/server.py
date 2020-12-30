#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:cch

import argparse
from sanic import Sanic
from sanic.response import json
from chatbot.generate.generate_dialogue import DialogueGenerator
import torch

# sys.path.append(os.getcwd())
app = Sanic("test sanic")

parser = argparse.ArgumentParser()
parser.add_argument("-mp",
                    "--model_path",
                    dest="model_path",
                    help="The GPT2 model path",
                    # required=True,
                    type=str,
                    default='/home/cch/PycharmProjects/nlg_chatbot/dialogue_model'
                    )

parser.add_argument("-p",
                    "--port",
                    dest="port",
                    help="The server port",
                    # required=True,
                    default=8000,
                    type=int)

parser.add_argument("-uc",
                    "--use_cuda",
                    dest="use_cuda",
                    help="use cuda in the process",
                    # required=True,
                    type=bool
                    )
parser.add_argument("-vp",
                    "--vocab_path",
                    dest="vocab_path",
                    help="give the vacab path",
                    # required=True,
                    type=str,
                    default="/home/cch/PycharmProjects/nlg_chatbot/vocabulary/vocab_small.txt"
                    )

args = parser.parse_args()

generator = DialogueGenerator(model_path=args.model_path,
                              device=torch.device(
                                  "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"),
                              vocab_path=args.vocab_path
                              )


@app.route("/generate/dialogue")
async def gen_txt(request):
    req = request.json
    history = req["history"]
    text = req["text"]
    gen_text, gen_history = generator.chat(text=text, history=history)
    return json({
        "text": gen_text,
        "history": gen_history
    })


def main():
    app.run(host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
