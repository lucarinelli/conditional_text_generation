from dataset_utils import *
import re


def download_wikitext_dataset(data_path=DATA_PATH):
    # download only if don't have it already
    if not os.path.isdir(os.path.join(data_path,"wikitext-103")):
        print("Downloading Wikitext dataset...")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        subprocess.run(["wget","-P", data_path, "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"])
        subprocess.run(["unzip", "-d", data_path, os.path.join(data_path,"wikitext-103-v1.zip")])

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

def write_wikitext_json_chunks(dataset, split, data_path, chunk_size, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type):
    chunks = [dataset[start:min(start+chunk_size,len(dataset))] for start in range(0, len(dataset), chunk_size)]
    pool = mp.Pool(processes=8)
    pool.map(process_chunk_json, [(chunk_n, chunk_items, data_path, split, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type, "wikitext_captions") for chunk_n, chunk_items in enumerate(chunks)])

def write_wikitext_txt_chunks(dataset, split, data_path, chunk_size, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type):
    chunks = [dataset[start:min(start+chunk_size,len(dataset))] for start in range(0, len(dataset), chunk_size)]
    pool = mp.Pool(processes=8)
    pool.map(process_chunk_txt, [(chunk_n, chunk_items, data_path, split, use_control_codes, control_codes_powerset, max_control_codes_per_caption, control_codes_type, "wikitext_captions") for chunk_n, chunk_items in enumerate(chunks)])

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


def encode_wikitext(examples, tokenizer):
    encoded = tokenizer(examples['caption'], truncation=True, max_length=64, padding="max_length")
    encoded['labels'] = encoded['input_ids']
    return encoded

def encode_wikitext_dataset(dataset, tokenizer): 
    return dataset.map(lambda x: encode_wikitext(x, tokenizer), batched=True)

def encode_and_format_wikitext_dataset(dataset, tokenizer):
    encoded = encode_wikitext_dataset(dataset, tokenizer)
    encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return encoded