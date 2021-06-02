import os
import sys
import subprocess  # to run sh commands
import json
import multiprocessing as mp

DATA_PATH="./data"


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
