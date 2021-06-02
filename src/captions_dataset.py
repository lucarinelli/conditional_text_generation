'''
Everything related COCO captions loading and preprocessing
'''

from enum import Enum
import os
import sys
import subprocess  # to run sh commands
import json
from torch.utils.data import Dataset
import torch
from pathlib import Path
from itertools import chain, combinations, groupby
from datasets import load_dataset, Dataset
import glob
import multiprocessing as mp
import re

DATA_PATH="./data"

class DatasetType(Enum):
    TRAIN = 1
    EVAL = 2

def download_annotations_dataset(data_path=DATA_PATH):
    # download only if don't have it already
    if not os.path.isdir(os.path.join(data_path,"annotations")):
        print("Downloading COCO dataset...")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        subprocess.run(["wget","-P", data_path, "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"])
        subprocess.run(["unzip", "-d", data_path, os.path.join(data_path,"annotations_trainval2017.zip")])

def download_wikitext_dataset(data_path=DATA_PATH):
    # download only if don't have it already
    if not os.path.isdir(os.path.join(data_path,"wikitext-103")):
        print("Downloading Wikitext dataset...")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        subprocess.run(["wget","-P", data_path, "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"])
        subprocess.run(["unzip", "-d", data_path, os.path.join(data_path,"wikitext-103-v1.zip")])

def map_and_join_dataset(data_instances, data_captions, use_supercategories=True, use_categories=False, captions_per_image_id = 5):
    if not use_categories and not use_supercategories:
        print("One of categories and supercategories has to be used!")
        sys.exit()

    categories_data_dict = dict(map(lambda c: (c["id"], c), data_instances["categories"])) # <category_id, category>
    annotations_data_mapped = map(lambda c: (c["image_id"], c), data_instances["annotations"]) # <image_id, annotation>
    annotations_data_dict = {}
    
    for a in annotations_data_mapped:
        if a[0] in annotations_data_dict:
            annotations_data_dict[a[0]] += [a[1]]
        else:
            annotations_data_dict[a[0]] = [a[1]]

    
    captions_data_list = list(map(lambda c: (c["image_id"], c), data_captions["annotations"]))
    captions_data_dict = dict()
    for image_id, image_captions in groupby(sorted(captions_data_list, key=lambda x: x[0]), lambda x: x[0]): #<image_id, list(caption)>
      image_captions_dict = dict()
      for caption in image_captions:
        image_captions_dict[caption[1]["id"]]=caption[1]
      captions_data_dict[image_id]=image_captions_dict

    dataset = []
    control_codes_dict = {}
    no_category_counter = 0
    references_dict = {}

    captions_data_dict_filtered = {}
    discarded = 0

    

    for image_id, captions in captions_data_dict.items():
        if len(captions) >= captions_per_image_id:
            discarded += max(0, len(captions) - captions_per_image_id)
            captions_data_dict_filtered[image_id] = dict(list(captions.items())[0:4])
        else:
            discarded += len(captions)

    print("Discarded: "+str(discarded))

    for image_id, captions in captions_data_dict_filtered.items():
        #references_dict[image_id] = list(map(lambda x: x[1]["caption"], captions.items()))
        for _, caption in captions.items():
          item = {"caption": caption["caption"], "categories": [], "image_id": image_id}
          if image_id in annotations_data_dict:
              tmp_categories_dict = {}
              for a in annotations_data_dict[image_id]:
                  category_name = categories_data_dict[a["category_id"]]["name"]
                  supercategory_name = categories_data_dict[a["category_id"]]["supercategory"]
                  if use_categories:
                      tmp_categories_dict[category_name] = 1
                      control_codes_dict[category_name] = 1
                  if use_supercategories:
                    tmp_categories_dict[supercategory_name] = 1
                    control_codes_dict[supercategory_name] = 1
              item["categories"]=list(tmp_categories_dict.keys())
          if len(item["categories"])==0:
              no_category_counter += 1
          else: 
            dataset += [item]
            if image_id in references_dict:
              references_dict[image_id] += [caption["caption"]]
            else:
              references_dict[image_id] = [caption["caption"]]

    print("There are "+str(no_category_counter)+" captions without a category")
    return dataset, references_dict, list(control_codes_dict.keys())

def parse_wikitext_dataset(file_name):
    dataset = {}
    control_codes = set()
    with open(file_name , "r") as f:
        currentCategory = None
        no_categories_counter = 0
        while True:
            line = f.readline()

            if not line:
                break
            trimmed = line.strip()

            if not trimmed:
                continue

            if re.search("^( =)[a-zA-Z ]+(= )$",line) is not None:
                continue #skip the article header
            elif re.search("^( =){2,}[a-zA-Z ]+(= ){2,}$", line) is not None:
                currentCategory = re.sub("( =){2,}|(= ){2,}", "", trimmed)
                control_codes.add(currentCategory)
            else:
                if currentCategory is None:
                    no_categories_counter += 1
                    continue
                '''if not currentCategory in dataset.keys():
                dataset[currentCategory] = set()
                dataset[currentCategory].add(trimmed)'''
                if not trimmed in dataset.keys():
                    dataset[trimmed] = set()
                dataset[trimmed].add(currentCategory)

    pretty_dataset = []
    for caption in dataset:
        pretty_dataset += [{"caption": caption, "categories": list(dataset[caption])}]
        
    print("There are "+str(no_categories_counter)+" captions without a category")
    return pretty_dataset, list(control_codes)


def load_or_setup_dataset(data_path=DATA_PATH, split='train', use_supercategories=True, use_categories=False, force_dataset_update=False, captions_per_image_id = 5):
    if not split in ['train', 'val']:
        print("Unknown split: "+split)
        sys.exit()
    if not force_dataset_update and os.path.isfile(os.path.join(data_path, "dataset_with_ctrl_"+split+".json")):
        print ("Dataset json file, loading dataset...")
        with open(os.path.join(data_path, "dataset_with_ctrl_"+split+".json"), "r") as read_file:
            dataset = json.load(read_file)
        with open(os.path.join(data_path, "control_codes_"+split+".json"), "r") as read_file:
            control_codes = json.load(read_file)
        with open(os.path.join(data_path, "references_"+split+".json"), "r") as read_file:
            references_dict = json.load(read_file)
    else:
        print ("Dataset json file does not exist, creating dataset from scratch...")
        download_annotations_dataset(data_path=data_path)
        with open(os.path.join(data_path,"annotations/instances_"+split+"2017.json"), "r") as read_file:
            data_instances = json.load(read_file)

        with open(os.path.join(data_path,"annotations/captions_"+split+"2017.json"), "r") as read_file:
            data_captions = json.load(read_file)

        dataset, references_dict, control_codes = map_and_join_dataset(data_instances, data_captions, use_supercategories, use_categories, captions_per_image_id)

        with open(os.path.join(data_path,"control_codes_"+split+".json"), 'w') as outfile:
            json.dump(control_codes, outfile)

        with open(os.path.join(data_path,"references_"+split+".json"), 'w') as outfile:
            json.dump(references_dict, outfile)
        
        with open(os.path.join(data_path,"dataset_with_ctrl_"+split+".json"), 'w') as outfile:
            json.dump(dataset, outfile)
    return dataset, references_dict, control_codes

def load_or_setup_wikitext_dataset(data_path=DATA_PATH, split='train', force_dataset_update=False):
    if not split in ['train', 'val', 'test']:
        print("Unknown split: "+split)
        sys.exit()

    if not force_dataset_update and os.path.isfile(os.path.join(data_path, "dataset_with_wikitext_"+split+".json")):
        print ("Dataset json file, loading dataset...")
        with open(os.path.join(data_path, "dataset_with_wikitext_"+split+".json"), "r") as read_file:
            dataset = json.load(read_file)
        with open(os.path.join(data_path, "control_codes_with_wikitext_"+split+".json"), "r") as read_file:
            control_codes = json.load(read_file)
        
    else:
        print("Dataset json file does not exist, creating dataset from scratch...")

        download_wikitext_dataset(data_path=data_path)

        dataset, control_codes = parse_wikitext_dataset(os.path.join(data_path,"wikitext-103/wiki."+split+".tokens"))

        with open(os.path.join(data_path,"control_codes_with_wikitext_"+split+".json"), 'w') as outfile:
            json.dump(control_codes, outfile)
        
        with open(os.path.join(data_path,"dataset_with_wikitext_"+split+".json"), 'w') as outfile:
            json.dump(dataset, outfile)

    return dataset, control_codes

def powerset(iterable, max_size=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if max_size is None:
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    else:
        return chain.from_iterable(combinations(s, r) for r in range(min(max_size, len(s)+1)))

def process_chunk_json(parameters):
    chunk_number = parameters[0]
    chunk_items = parameters[1]
    data_path = parameters[2]
    split = parameters[3]
    use_control_codes = parameters[4]
    control_codes_powerset = parameters[5]
    max_control_codes_per_caption = parameters[6]
    control_codes_type = parameters[7]
    chunk_prefix = parameters[8]

    json_file = os.path.join(data_path, chunk_prefix+"_"+split+"_"+str(chunk_number)+".json")
    captions_array_for_json = []
    for item in chunk_items:
        if use_control_codes:
            if control_codes_powerset:
                control_codes_combinations = powerset(item['categories'], max_control_codes_per_caption)
            else:
                control_codes_combinations = [item['categories']]
        else:
            control_codes_combinations = [[]]
        for control_codes_combination in control_codes_combinations:
            pre_control_codes_string=""
            for category in sorted(control_codes_combination):
                if control_codes_type == "special_token":
                    pre_control_codes_string+="<CTRL:"+category.replace(" ","_")+">"
                elif control_codes_type == "separators":
                    pre_control_codes_string+=category+", "
                else:
                    print("ERROR: wrong control code type")
                    return -1  # TODO here we could fail better
            captions_array_for_json += [{"caption": pre_control_codes_string+'<|endoftext|>'+item["caption"]+'<|endoftext|>',"image_id": item["image_id"]}]
    with open(json_file, 'w') as captions_json:
        json.dump({"data": captions_array_for_json}, captions_json)

def write_json_chunks(dataset, split, data_path, chunk_size, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type):
    chunks = [dataset[start:min(start+chunk_size,len(dataset))] for start in range(0, len(dataset), chunk_size)]
    pool = mp.Pool(processes=8)
    pool.map(process_chunk_json, [(chunk_n, chunk_items, data_path, split, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type, "captions") for chunk_n, chunk_items in enumerate(chunks)])

def write_wikitext_json_chunks(dataset, split, data_path, chunk_size, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type):
    chunks = [dataset[start:min(start+chunk_size,len(dataset))] for start in range(0, len(dataset), chunk_size)]
    pool = mp.Pool(processes=8)
    pool.map(process_chunk_json, [(chunk_n, chunk_items, data_path, split, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type, "wikitext_captions") for chunk_n, chunk_items in enumerate(chunks)])

def process_chunk_txt(parameters):
    chunk_number = parameters[0]
    chunk_items = parameters[1]
    data_path = parameters[2]
    split = parameters[3]
    use_control_codes = parameters[4]
    control_codes_powerset = parameters[5]
    max_control_codes_per_caption = parameters[6]
    control_codes_type = parameters[7]
    chunk_prefix = parameters[8]

    json_file = os.path.join(data_path, chunk_prefix+"_"+split+"_"+str(chunk_number)+".txt")
    captions_array_for_json = []
    with open(json_file, 'w') as captions_txt:
        for item in chunk_items:
            if use_control_codes:
                if control_codes_powerset:
                    control_codes_combinations = powerset(item['categories'], max_control_codes_per_caption)
                else:
                    control_codes_combinations = [item['categories']]
            else:
                control_codes_combinations = [[]]
            for control_codes_combination in control_codes_combinations:
                pre_control_codes_string=""
                for category in sorted(control_codes_combination):
                    if control_codes_type == "special_token":
                        pre_control_codes_string+="<CTRL:"+category.replace(" ","_")+">"
                    elif control_codes_type == "separators":
                        pre_control_codes_string+=category+", "
                    else:
                        print("ERROR: wrong control code type")
                        return -1  # TODO here we could fail better
                captions_txt.write(pre_control_codes_string+'<|endoftext|>'+item["caption"]+'<|endoftext|>\n')

def write_txt_chunks(dataset, split, data_path, chunk_size, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type):
    chunks = [dataset[start:min(start+chunk_size,len(dataset))] for start in range(0, len(dataset), chunk_size)]
    pool = mp.Pool(processes=8)
    pool.map(process_chunk_json, [(chunk_n, chunk_items, data_path, split, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type, "captions") for chunk_n, chunk_items in enumerate(chunks)])

def write_wikitext_txt_chunks(dataset, split, data_path, chunk_size, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type):
    chunks = [dataset[start:min(start+chunk_size,len(dataset))] for start in range(0, len(dataset), chunk_size)]
    pool = mp.Pool(processes=8)
    pool.map(process_chunk_json, [(chunk_n, chunk_items, data_path, split, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type, "wikitext_captions") for chunk_n, chunk_items in enumerate(chunks)])

def get_dataset(exp_pars, data_path=DATA_PATH):
    dataset_train, _, categories = load_or_setup_dataset(data_path=data_path, split="train", use_supercategories=exp_pars.use_supercategories, use_categories=exp_pars.use_categories, force_dataset_update=exp_pars.force_dataset_update, captions_per_image_id=exp_pars.captions_per_image_id)
    dataset_val, references_validation, _ = load_or_setup_dataset(data_path=data_path, split="val", use_supercategories=exp_pars.use_supercategories, use_categories=exp_pars.use_categories, force_dataset_update=exp_pars.force_dataset_update, captions_per_image_id=exp_pars.captions_per_image_id)

    print("There are "+str(len(dataset_train))+" captions considered in total (train)")
    print("There are "+str(len(dataset_val))+" captions considered in total (val)")

    print("The following "+str(len(categories))+" categories are present in the dataset:")
    print(categories)

    control_codes = []
    if exp_pars.use_control_codes and exp_pars.control_codes_type == "special_token":
        for category in categories:
            control_codes += ["<CTRL:"+category.replace(" ","_")+">"]

        print("Processed control codes:")
        print(control_codes)

    chunk_size = exp_pars.chunk_size_json_mp
    print("Writing dataset to json chunks")
    write_json_chunks(dataset_train, "train", data_path, chunk_size, exp_pars.use_control_codes, exp_pars.control_codes_powerset, exp_pars.max_control_codes_per_caption, exp_pars.control_codes_type)
    write_json_chunks(dataset_val, "val", data_path, chunk_size, exp_pars.use_control_codes, exp_pars.control_codes_powerset, exp_pars.max_control_codes_per_caption, exp_pars.control_codes_type)

    # Write txt files for tokenizer training if necessary
    if not exp_pars.pretrained:
        print("Writing txt files for tokenizer")
        write_txt_chunks(dataset_train, "train", data_path, chunk_size, False, False, 0, None)

    print("Load dataset in HuggingFace datasets")

    dataset_train, dataset_val = load_dataset('json', data_files={'train': glob.glob(os.path.join(data_path, 'captions_train_*.json')), 'val': glob.glob(os.path.join(data_path, 'captions_val_*.json'))}, split=['train', 'val'], field="data")
    
    print("Post-processed dataset has "+str(len(dataset_train))+" train elements and "+str(len(dataset_val))+" validation elements")

    if exp_pars.limited_run: # shuffle and cut the datasets
        dataset_train = dataset_train.shuffle(exp_pars.random_seed).select(range(exp_pars.max_train_set_len))
        dataset_val = dataset_val.shuffle(exp_pars.random_seed).select(range(exp_pars.max_val_set_len))
        print("We take only a small part of that: "+str(len(dataset_train))+" train elements and "+str(len(dataset_val))+" validation elements")
    else: # just shuffle them
        dataset_train = dataset_train.shuffle(exp_pars.random_seed)
        dataset_val = dataset_val.shuffle(exp_pars.random_seed)
        print("Train elements: "+str(len(dataset_train))+"\nValidation elements: "+str(len(dataset_val)))

    
    return dataset_train, dataset_val, control_codes, references_validation

def get_wikitext_dataset(exp_pars, data_path=DATA_PATH):
    dataset_train, categories = load_or_setup_wikitext_dataset(data_path=data_path, split="train", force_dataset_update=exp_pars.force_dataset_update)
    dataset_val, _ = load_or_setup_wikitext_dataset(data_path=data_path, split="val", force_dataset_update=exp_pars.force_dataset_update)

    print("There are "+str(len(dataset_train))+" captions considered in total (train)")
    print("There are "+str(len(dataset_val))+" captions considered in total (val)")

    print("The following "+str(len(categories))+" categories are present in the dataset:")
    print(categories)

    control_codes = []
    if exp_pars.use_control_codes and exp_pars.control_codes_type == "special_token":
        for category in categories:
            control_codes += ["<CTRL:"+category.replace(" ","_")+">"]

        print("Processed control codes:")
        print(control_codes)

    chunk_size = exp_pars.chunk_size_json_mp
    print("Writing dataset to json chunks")
    write_wikitext_json_chunks(dataset_train, "train", data_path, chunk_size, exp_pars.use_control_codes, exp_pars.control_codes_powerset, exp_pars.max_control_codes_per_caption, exp_pars.control_codes_type)
    write_wikitext_json_chunks(dataset_val, "val", data_path, chunk_size, exp_pars.use_control_codes, exp_pars.control_codes_powerset, exp_pars.max_control_codes_per_caption, exp_pars.control_codes_type)

    # Write txt files for tokenizer training if necessary
    if not exp_pars.pretrained:
        print("Writing txt files for tokenizer")
        write_wikitext_txt_chunks(dataset_train, "train", data_path, chunk_size, False, False, 0, None)

    print("Load dataset in HuggingFace datasets")

    dataset_train, dataset_val = load_dataset('json', data_files={'train': glob.glob(os.path.join(data_path, 'wikitext_captions_train_*.json')), 'val': glob.glob(os.path.join(data_path, 'wikitext_captions_val_*.json'))}, split=['train', 'val'], field="data")
    
    print("Post-processed dataset has "+str(len(dataset_train))+" train elements and "+str(len(dataset_val))+" validation elements")

    if exp_pars.limited_run: # shuffle and cut the datasets
        dataset_train = dataset_train.shuffle(exp_pars.random_seed).select(range(exp_pars.max_train_set_len))
        dataset_val = dataset_val.shuffle(exp_pars.random_seed).select(range(exp_pars.max_val_set_len))
        print("We take only a small part of that: "+str(len(dataset_train))+" train elements and "+str(len(dataset_val))+" validation elements")
    else: # just shuffle them
        dataset_train = dataset_train.shuffle(exp_pars.random_seed)
        dataset_val = dataset_val.shuffle(exp_pars.random_seed)
        print("Train elements: "+str(len(dataset_train))+"\nValidation elements: "+str(len(dataset_val)))

    
    return dataset_train, dataset_val, control_codes

def get_tokenizer(exp_pars, control_codes = []):
    if exp_pars.pretrained:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained(exp_pars.model)
    else:
        from tokenizers import ByteLevelBPETokenizer
        from transformers import PreTrainedTokenizerFast        

        # Initialize a tokenizer
        tokenizer = ByteLevelBPETokenizer()
        print("Training tokenizer")
        tokenizer.train(files=glob.glob(os.path.join(exp_pars.data_path, 'captions_train_*.txt')), vocab_size=52000, min_frequency=2, special_tokens=[
            "<|endoftext|>",
        ])
        # Save the trained tokenizer
        tokenizer.save("byte-level-BPE.tokenizer.json")
        # Load it using transformers

        from typing import Optional, Tuple

        class PreTrainedTokenizerFastMod(PreTrainedTokenizerFast):
            def save_vocabulary(self, save_directory: str = exp_pars.data_path, filename_prefix: Optional[str] = None) -> Tuple[str]:
                files = self._tokenizer.model.save(save_directory, name=filename_prefix)
                return tuple(files)

        tokenizer = PreTrainedTokenizerFastMod(tokenizer_file="byte-level-BPE.tokenizer.json")
        tokenizer.add_special_tokens({'eos_token': "<|endoftext|>"})

    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer before added special tokens "+str(len(tokenizer)))

    if exp_pars.use_control_codes and exp_pars.control_codes_type == "special_token":
        special_tokens_dict = {'additional_special_tokens': control_codes}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("added "+str(num_added_toks)+" tokens to the pretrained tokenizer")

    return tokenizer

def encode(examples, tokenizer):
    encoded = tokenizer(examples['caption'], truncation=True, max_length=64, padding="max_length")
    encoded['labels'] = encoded['input_ids']
    encoded['image_id'] = examples['image_id']
    return encoded

def encode_wikitext(examples, tokenizer):
    encoded = tokenizer(examples['caption'], truncation=True, max_length=64, padding="max_length")
    encoded['labels'] = encoded['input_ids']
    return encoded

def encode_dataset(dataset, tokenizer): 
    return dataset.map(lambda x: encode(x, tokenizer), batched=True)

def encode_wikitext_dataset(dataset, tokenizer): 
    return dataset.map(lambda x: encode_wikitext(x, tokenizer), batched=True)

def encode_and_format_dataset(dataset, dataset_Type: DatasetType, tokenizer):
    encoded = encode_dataset(dataset, tokenizer)
    if DatasetType.TRAIN == dataset_Type:
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    elif DatasetType.EVAL == dataset_Type:
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'image_id'])
    return encoded

def encode_and_format_wikitext_dataset(dataset, tokenizer):
    encoded = encode_dataset(dataset, tokenizer)
    encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return encoded