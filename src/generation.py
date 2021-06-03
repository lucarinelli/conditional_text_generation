import torch
from enum import Enum

class ControlCodeType(Enum):
    SEPARATOR = 1
    SPECIAL_TOKEN = 2

class Generator():
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer


  def generate(self, control_codes, type: ControlCodeType, input, max_len, num_return_sequences=3):
    joiner = None
    if type == ControlCodeType.SEPARATOR:
        joiner = ", "
    else:
        joiner = ""
        control_codes = map(lambda x: "<CTRL:"+x+">", control_codes)
    
    ctrl = joiner.join(control_codes)
    prompt = ctrl + input
            
    generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    self.model.eval()
    sample_outputs = self.model.generate(generated, 
                                    do_sample=True,   
                                    max_length=max_len,
                                    top_k=30,                                 
                                    top_p=0.7,     
                                    temperature=0.9,
                                    repetition_penalty=2.0,
                                    num_return_sequences= num_return_sequences
                                    )

    for i, sample_output in enumerate(sample_outputs):
        text = self.tokenizer.decode(sample_output, skip_special_tokens=True)
        text = text.replace(ctrl, "")
        print("{}: {}\n\n".format(i+1,  text))