import argparse
import os
from src.generation import *

""" input, 
      max_len, num_return_sequences=3,
      top_k=30, top_p=0.7, 
      temperature=0.9, repetition_penalty=2.0): """

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, required=False,
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
parser.add_argument('--control_codes_type',type=ControlCodeType,
                                        help='Control codes type', choices=list(ControlCodeType) )

parser.add_argument('--num_returned_sequences',type=int,default=3,
                                        help='the number of sentences the model will generate' )

args = parser.parse_args()

artifact_dir = args.model_dir

if artifact_dir is None or not os.path.isdir(artifact_dir):
    artifact_dir = "./artifacts/model-3152aoah-v0"
    if not os.path.isdir("./artifacts/model-3152aoah:v0"): 
        os.environ["WANDB_API_KEY"] = "92907f006616f5c5d84bf6f28f4ab8f6220b5ea1"
        import wandb
        run = wandb.init()
        artifact = run.use_artifact('polito_aiml2021_textgen/dstilgpt2-specialtoken/model-3152aoah:v0', type='model')
        artifact_dir = artifact.download()



from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained(artifact_dir).cuda()
tokenizer = GPT2TokenizerFast.from_pretrained(artifact_dir)

generator = Generator(model,tokenizer)

control_codes = args.control_codes.split(", ")

generator.generate(control_codes, args.control_codes_type, args.input, args.max_len,
                    args.num_returned_sequences, args.top_k, args.top_p, args.temperature, 
                    args.repetition_penalty )