import torch
from enum import Enum
import re

class ControlCodeType(Enum):
    SEPARATOR = 1
    SPECIAL_TOKEN = 2

class Generator():
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer


  def generate(self, control_codes, type: ControlCodeType, input, 
      max_len, num_return_sequences=3,
      do_sample=True, top_k=30, top_p=0.7, 
      temperature=0.9, repetition_penalty=2.0):

    joiner = None
    if type == ControlCodeType.SEPARATOR:
        joiner = ", "
    else:
        joiner = ""
        control_codes = list(map(lambda x: "<CTRL:"+x+">", control_codes))
    
    ctrl = joiner.join(control_codes)
    prompt = ctrl + input
            
    generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
    
    if torch.cuda.is_available():
      device = torch.device("cuda")
    else :
      device = torch.device("cpu")


    generated = generated.to(device)

    self.model.eval()
    sample_outputs = self.model.generate(generated, 
                                    do_sample=do_sample,   
                                    max_length=max_len,
                                    top_k=top_k,                                 
                                    top_p=top_p,     
                                    temperature=temperature,
                                    repetition_penalty=repetition_penalty,
                                    num_return_sequences= num_return_sequences
                                    )
    for i, sample_output in enumerate(sample_outputs):
        text = self.tokenizer.decode(sample_output, skip_special_tokens=True)
        for control_code in control_codes:
          text = re.sub("^("+joiner+"){0,1}"+control_code, "", text)
        print("{}: {}\n\n".format(i+1,  text))