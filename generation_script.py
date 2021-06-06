import argparse
import os
from src.generation import *
from enum import Enum

class Models(Enum):
    ST = {"code_type" : ControlCodeType.SPECIAL_TOKEN, 
        "name": "ST", 
        "url":  "polito_aiml2021_textgen/uncategorized/gpt2-specialtoken:v1",
        "folder" : ""}
    SEP = {"code_type": ControlCodeType.SEPARATOR, 
        "name" : "SEP" ,
        "url":"polito_aiml2021_textgen/gpt2-separators/model-8p9purad:v0",
        "folder": "" }
    ST_10F = {"code_type" : ControlCodeType.SPECIAL_TOKEN, 
        "name": "ST_10F", 
        "url":"polito_aiml2021_textgen/uncategorized/gpt2-specialtoken-10freezed:v0",
        "folder": ""}
    SEP_10F = {"code_type" : ControlCodeType.SEPARATOR, 
        "name": "SEP_10F",  
        "url":"polito_aiml2021_textgen/uncategorized/gpt2-TRUE_separators-10freezed:v0",
        "folder": ""}
    D_ST = {code_type : ControlCodeType.SPECIAL_TOKEN, 
        "name": "D_ST", 
        "url":"polito_aiml2021_textgen/distilgpt2-specialtoken/model-3b1lzdro:v0",
        "folder": "" }

def __str__(self):
        return self.value.name

""" input, 
      max_len, num_return_sequences=3,
      top_k=30, top_p=0.7, 
      temperature=0.9, repetition_penalty=2.0): """

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=Models, required=True, choices=list(Models),
                                        help='location of model checkpoint')


parser.add_argument('--input', type=str, default= "<|startoftext|>",
                                        help='The start of the sequence(s) the model will generate')

parser.add_argument('--max_len', type=int, default=16,
                                        help='number of tokens to generate')

parser.add_argument('--temperature', type=float, default=0.9,
                                        help='temperature for sampling distribution; 0 means greedy')

parser.add_argument('--top_k', type=int, default=30,
                                        help='topk value for sampling from the softmax distribution ; 0 means no topk preferred')

parser.add_argument('--repetition_penalty', type=float, default=2.0,
                                        help='repetition penalty for greedy sampling')
parser.add_argument('--top_p', type=int, default=0.7,
                                        help='print top-n candidates during generations; defaults to 0 which is no printing')                                  

parser.add_argument('--control_codes',type=str, default="",
                                        help='Control codes to be used during generation separated by ", "' )

parser.add_argument('--num_returned_sequences',type=int,default=3,
                                        help='the number of sentences the model will generate' )

args = parser.parse_args()

artifact_dir = args.model["folder"]

if not os.path.isdir(artifact_dir):
    os.environ["WANDB_API_KEY"] = "92907f006616f5c5d84bf6f28f4ab8f6220b5ea1"
    import wandb
    run = wandb.init()
    artifact = run.use_artifact(args.model.url, type='model')
    artifact_dir = artifact.download()



from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained(artifact_dir).cuda()
tokenizer = GPT2TokenizerFast.from_pretrained(artifact_dir)

generator = Generator(model,tokenizer)

control_codes = args.control_codes.split(", ")

generator.generate(control_codes, args.args.model["code_type"], args.input, args.max_len,
                    args.num_returned_sequences, args.top_k, args.top_p, args.temperature, 
                    args.repetition_penalty )