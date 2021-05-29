'''
Everything related COCO captions loading and preprocessing
'''

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

DATA_PATH="./data"

def download_annotations_dataset(data_path=DATA_PATH):
    # download only if don't have it already
    if not os.path.isdir(os.path.join(data_path,"annotations")):
        print("Downloading COCO dataset...")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        subprocess.run(["wget","-P", data_path, "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"])
        subprocess.run(["unzip", "-d", data_path, os.path.join(data_path,"annotations_trainval2017.zip")])

def map_and_join_dataset(data_instances, data_captions):
    if not experiment_parameters["use_categories"] and not experiment_parameters["use_supercategories"]:
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
        if len(captions) >= 5:
            discarded += max(0, len(captions) - 5)
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
                  if experiment_parameters["use_categories"]:
                      tmp_categories_dict[category_name] = 1
                      control_codes_dict[category_name] = 1
                  if experiment_parameters["use_supercategories"]:
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


    #TODO compute total of captions?

    print("There are "+str(no_category_counter)+" captions without a category")
    return dataset, references_dict, list(control_codes_dict.keys())

def load_or_setup_dataset(data_path=DATA_PATH, split='train'):
    if not split in ['train', 'val']:
        print("Unknown split: "+split)
        sys.exit()
    if not experiment_parameters["force_dataset_update"] and os.path.isfile(os.path.join(data_path, "dataset_with_ctrl_"+split+".json")):
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

        dataset, references_dict, control_codes = map_and_join_dataset(data_instances, data_captions)

        with open(os.path.join(data_path,"control_codes_"+split+".json"), 'w') as outfile:
            json.dump(control_codes, outfile)

        with open(os.path.join(data_path,"references_"+split+".json"), 'w') as outfile:
            json.dump(references_dict, outfile)
        
        with open(os.path.join(data_path,"dataset_with_ctrl_"+split+".json"), 'w') as outfile:
            json.dump(dataset, outfile)
    return dataset, references_dict, control_codes

import multiprocessing as mp

def powerset(iterable, max_size=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if max_size is None:
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    else:
        return chain.from_iterable(combinations(s, r) for r in range(min(max_size, len(s)+1)))

def process_chunk(chunk):
    chunk_number = chunk[0]
    chunk_items = chunk[1]
    data_path = chunk[2]
    split = chunk[3]
    json_file = os.path.join(data_path, "captions_"+split+"_"+str(chunk_number)+".json")
    captions_array_for_json = []
    for item in chunk_items:
        if experiment_parameters["use_control_codes"]:
            if experiment_parameters["use_control_codes_powerset"]:
                control_codes_combinations = powerset(item['categories'], experiment_parameters["max_control_codes_per_caption"])
            else:
                control_codes_combinations = [item['categories']]
        else:
            control_codes_combinations = [[]]
        for control_codes_combination in control_codes_combinations:
            pre_control_codes_string=""
            for category in sorted(control_codes_combination):
                if experiment_parameters["control_codes_type"] == "special_token":
                    pre_control_codes_string+="<CTRL:"+category.replace(" ","_")+">"
                elif experiment_parameters["control_codes_type"] == "separators":
                    pre_control_codes_string+=category+", "
                else:
                    print("ERROR: wrong control code type")
                    return -1  # TODO here we could fail better
            captions_array_for_json += [{"caption": pre_control_codes_string+'<|endoftext|>'+item["caption"]+'<|endoftext|>',"image_id": item["image_id"]}]
    with open(json_file, 'w') as captions_json:
        json.dump({"data": captions_array_for_json}, captions_json)


def write_json_chunks(dataset, split, data_path, chunk_size):
    chunks = [dataset[start:min(start+chunk_size,len(dataset))] for start in range(0, len(dataset), chunk_size)]
    pool = mp.Pool(processes=8)
    pool.map(process_chunk, [(chunk_n, chunk_items, data_path, split) for chunk_n, chunk_items in enumerate(chunks)])


def load_our_dataset(data_path, split):
    dataset_train, _, categories = load_or_setup_dataset(data_path=data_path, split="train")
    dataset_val, references, _ = load_or_setup_dataset(data_path=data_path, split="val")

    print("There are "+str(len(dataset_train))+" captions considered in total (train)")
    print("There are "+str(len(dataset_val))+" captions considered in total (val)")

    print("The following "+str(len(categories))+" categories are present in the dataset:")
    print(categories)

    if experiment_parameters["use_control_codes"] and experiment_parameters["control_codes_type"] == "special_token":
        control_codes = []
        for category in categories:
            control_codes += ["<CTRL:"+category.replace(" ","_")+">"]

        print("Processed control codes:")
        print(control_codes)

    chunk_size = experiment_parameters["chunk_size"]
    write_json_chunks(dataset_train, "train", data_path, chunk_size)
    write_json_chunks(dataset_val, "val", data_path, chunk_size)
  
    dataset_train, dataset_val = load_dataset('json', data_files={'train': glob.glob('./data/captions_train_*.json'), 'val': glob.glob('./data/captions_val_*.json')}, split=['train', 'val'], field="data")
    return dataset_train, dataset_val