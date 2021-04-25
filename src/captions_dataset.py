import os
import sys
import subprocess  # to run sh commands
import json
from torch.utils.data import Dataset
import torch
from pathlib import Path

DATA_PATH="../data"

def download_annotations_dataset(data_path=DATA_PATH):
    # download only if don't have it already
    if not os.path.isdir(os.path.join(data_path,"annotations")):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        subprocess.run(["wget","-P", data_path, "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"])
        subprocess.run(["unzip", os.path.join(data_path,"annotations_trainval2017.zip")])

def map_and_join_dataset(data_instances, data_captions):
    categories_data_dict = dict(map(lambda c: (c["id"], c), data_instances["categories"]))
    annotations_data_mapped = map(lambda c: (c["image_id"], c), data_instances["annotations"])
    annotations_data_dict = {}
    for a in annotations_data_mapped:
        if a[0] in annotations_data_dict:
            annotations_data_dict[a[0]] += [a[1]]
        else:
            annotations_data_dict[a[0]] = [a[1]]
    captions_data_dict = dict(map(lambda c: (c["image_id"], c), data_captions["annotations"]))

    dataset = []
    control_codes_dict = {}
    no_category_counter = 0

    for c in captions_data_dict.items():
        item = {"caption": c[1]["caption"], "categories": []}
        if c[1]["image_id"] in annotations_data_dict:
            tmp_categories_dict = {}
            for a in annotations_data_dict[c[1]["image_id"]]:
                category_name = categories_data_dict[a["category_id"]]["name"]
                supercategory_name = categories_data_dict[a["category_id"]]["supercategory"]
                tmp_categories_dict[category_name] = 1
                tmp_categories_dict[supercategory_name] = 1
                control_codes_dict[category_name] = 1
                control_codes_dict[supercategory_name] = 1
            item["categories"]=list(tmp_categories_dict.keys())
        if len(item["categories"])==0:
            no_category_counter += 1
        dataset += [item]

    print("There are "+str(no_category_counter)+" captions without a category")
    return dataset, list(control_codes_dict.keys())

def load_or_setup_dataset(data_path=DATA_PATH, split='train'):
    if not split in ['train', 'val']:
        print("Unknown split: "+split)
        sys.exit()
    if os.path.isfile(os.path.join(data_path, "dataset_with_ctrl_"+split+".json")):
        print ("Dataset json file, loading dataset...")
        with open(os.path.join(data_path, "dataset_with_ctrl_"+split+".json"), "r") as read_file:
            dataset = json.load(read_file)
        with open(os.path.join(data_path, "control_codes_"+split+".json"), "r") as read_file:
            control_codes = json.load(read_file)
    else:
        print ("Dataset json file does not exist, creating dataset from scratch...")
        download_annotations_dataset(data_path=data_path)
        with open(os.path.join(data_path,"annotations/instances_"+split+"2017.json"), "r") as read_file:
            data_instances = json.load(read_file)

        with open(os.path.join(data_path,"annotations/captions_"+split+"2017.json"), "r") as read_file:
            data_captions = json.load(read_file)

        dataset, control_codes = map_and_join_dataset(data_instances, data_captions)

        with open(os.path.join(data_path,"control_codes_"+split+".json"), 'w') as outfile:
            json.dump(control_codes, outfile)
        
        with open(os.path.join(data_path,"dataset_with_ctrl_"+split+".json"), 'w') as outfile:
            json.dump(dataset, outfile)
    return dataset, control_codes

def write_captions_txt(dataset, data_path=DATA_PATH, split='train'):
    if not split in ['train', 'val']:
        print("Unknown split: "+split)
        sys.exit()
    txt_file = os.path.join(data_path, "captions_"+split+".txt")
    if os.path.isfile(txt_file):
        print("txt file already exists, nothing to do here...")
    else:
        with open(txt_file, 'w') as captions_txt:
            for item in dataset:
                pre_control_codes_string=""
                for category in item['categories']:
                    pre_control_codes_string+="<CTRL:"+category.replace(" ","_")+">"
                captions_txt.write(pre_control_codes_string+" "+item['caption']+'\n')

class CaptionsDataset(Dataset):

    def __init__(self, data_path=DATA_PATH, split = "train", tokenizer=None):
        if not split in ['train', 'val']:
            print("Unknown split, use 'train' or 'val'")
            sys.exit()

        self.data_path = data_path
        self.split = split

        dataset, categories = load_or_setup_dataset(split=split)
        self.dataset = dataset
        self.categories = categories

        print("There are "+str(len(self.dataset))+" captions in total ("+split+")")

        print("The following "+str(len(self.categories))+" categories are present in the dataset:")
        print(self.categories)

        self.control_codes = []
        for category in categories:
            self.control_codes += ["<CTRL:"+category.replace(" ","_")+">"]

        print("Processed control codes:")
        print(self.control_codes)

        print("Writing txt")
        write_captions_txt(self.dataset, split=split)

        self.input_ids = []
        self.attention_masks = []
        self.entries = []

        if not tokenizer is None:
            print("Using tokenizer provided as argument")
            self.tokenize(tokenizer)
        else:
            self.tokenizer = None
            print("No tokenizer provided, you will need to provide one later through the tokenize method!")

    def tokenize(self, tokenizer, load_tokenized=False):
        self.input_ids = []
        self.attention_masks = []
        self.entries = []
        self.tokenizer = tokenizer

        if load_tokenized and os.path.isfile(os.path.join(self.data_path, "entries_"+self.split+".pt")):
            print("Loading pretokenized dataset! ("+self.split+")")
            self.entries = torch.load(os.path.join(self.data_path, "entries_"+self.split+".pt"))
        else:
            print("Tokenizing dataset! ("+self.split+")")
            for item in self.dataset:
                pre_control_codes_string=""
                for category in item['categories']:
                    pre_control_codes_string+="<CTRL:"+category.replace(" ","_")+">"
                x = tokenizer(pre_control_codes_string+'<|startoftext|>'+item['caption']+'<|endoftext|>', truncation=True, max_length=256, padding="max_length")
                self.input_ids += torch.tensor([x.input_ids])
                self.attention_masks += torch.tensor([x.attention_mask])
                self.entries += [{"labels": torch.tensor([x.input_ids]), "input_ids": torch.tensor([x.input_ids]), "attention_mask": torch.tensor([x.attention_mask])}]
            print("Saving tokenized dataset! ("+self.split+")")
            torch.save(self.entries, os.path.join(self.data_path, "entries_"+self.split+".pt"))       

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return self.entries[i]