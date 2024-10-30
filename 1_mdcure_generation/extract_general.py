import argparse
from tqdm import tqdm
import pandas as pd
import re
import random
import os
import torch
import json

def get_query(instr: str, prompt_num: int):

    if prompt_num==1:
        answer = ""
        try: 
            query = instr.removeprefix("X:")
        except: 
            query = ""

    elif prompt_num>=2:
        try:
            break_idx = instr.index("\nAnswer:")
            query_with_header = instr[:break_idx].strip()
            answer_with_header = instr[break_idx:].strip()
            query = query_with_header.removeprefix("Instruction:")
            answer = answer_with_header.removeprefix("Answer:")
        except: 
            try: 
                break_idx = instr.index("\nA: ")
                query_with_header = instr[:break_idx].strip()
                answer_with_header = instr[break_idx:].strip()
                query = query_with_header.removeprefix("Q:")
                answer = answer_with_header.removeprefix("A:")
            except: 
                query = ""
                answer = ""

    if "Instruction:" in query or "Answer:" in answer:
        query = ""
        answer = ""

    return query.strip(), answer.strip()

def finalize_instr(
                  prompt_num: int,
                  prompt: str,
                  query: str, # pre-stripped when passed in!
                  answer: str,
                  docs_or_snippets=None, # list of source docs
                  ):

    # obtain documents & answers
    docs = list(docs_or_snippets)

    if args.prompt_num==1:
        prompt_prefix = """Snippets: '"""
        end_idx = prompt.index("'\nContext Paragraphs: ") 
        snippets = prompt[:end_idx].removeprefix(prompt_prefix)
        snippets = snippets.split("', '")
        finalized_answer = " ".join(snippets)
        if random.random()<0.5:
            finalized_answer = abstractify(finalized_answer)
    else: 
        finalized_answer = answer

    finalized_instr = "'"
    finalized_instr += docs[0]
    finalized_instr += "'\n\n'"
    finalized_instr += docs[1]
    finalized_instr += "'\n\n"
    finalized_instr += query

    return finalized_instr, finalized_answer

def main(args):
    
    output_dir = os.path.join(args.input_dir, "data_jsons")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prompt_dir = os.path.join(args.input_dir,"prompts")
    instr_dir = os.path.join(args.input_dir,"instructions")
    cluster_ids_dir = os.path.join(args.input_dir,"cluster_ids") 

    # sort alphanumerically so that names match in order of reference 
    all_prompt_files_train = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_prompt_files_valid = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_prompt_files_test = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_prompt_files = [all_prompt_files_train, all_prompt_files_valid, all_prompt_files_test]

    all_instr_files_train = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_instr_files_valid = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_instr_files_test = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_instr_files = [all_instr_files_train, all_instr_files_valid, all_instr_files_test]

    all_cluster_id_files_train = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_cluster_id_files_valid = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "valid" in f
            ], key=str.lower), key=len)
    all_cluster_id_files_test = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "test" in f
            ], key=str.lower), key=len)
    all_cluster_id_files = [all_cluster_id_files_train, all_cluster_id_files_valid,all_cluster_id_files_test]

    # iterate over train/val/test splits
    for split_idx, (prompt_split_files, instr_split_files, cluster_id_files) in enumerate(tqdm(zip(all_prompt_files, all_instr_files, all_cluster_id_files))):

        if instr_split_files==[] or split_idx<args.start_split_idx:
            continue

        # create json file to save data for current split
        one_prompt_pt = instr_split_files[0] # e.g., train.1.pt
        print("Filenames:",prompt_split_files, instr_split_files, cluster_id_files)
        split_json_file_name = one_prompt_pt[:one_prompt_pt.index(".")] + ".json" # e.g., train.1.pt -> train.json
        data_json_path = os.path.join(output_dir, split_json_file_name)

        print("########################")
        print("Working on",split_json_file_name)
        
        # extract all the instructons for current split
        for file_idx, instr_pt in enumerate(tqdm(instr_split_files)):
            
            # write current filename to log (in case need to pause/resume)
            f = open(os.path.join(output_dir, "num_slices_processed_so_far.txt"), "w")
            f.write("\nOn instruction/prompt file: "+str(instr_pt))
            f.write("\nCorresponding file_idx: "+str(file_idx))
            f.close()
            
            if args.start_file_idx <= file_idx < len(instr_split_files): ## NOTE: change len(...) to desired number N if want to only process N instruction/prompt files

                prompt_slice = torch.load(os.path.join(prompt_dir, instr_pt))
                instr_slice = torch.load(os.path.join(instr_dir, instr_pt))
                cluster_id_slice = torch.load(os.path.join(cluster_ids_dir, instr_pt))

                doc_dir = os.path.join(args.input_dir,"source_docs")
                source_doc_slice = torch.load(os.path.join(doc_dir, instr_pt))

                for example_idx, (prompt, instr, cluster_id) in enumerate(zip(prompt_slice, instr_slice, cluster_id_slice)):
                    
                    query, answer = get_query(instr.strip(), args.prompt_num)

                    finalized_instr, finalized_answer = finalize_instr(args.prompt_num, prompt, query, answer, source_doc_slice[example_idx]) 

                    # save data to json
                    if finalized_answer != "" and finalized_instr != "" and "snippet" not in finalized_answer and "snippet" not in finalized_instr:
                        data = {"instruction": finalized_instr, "answer": finalized_answer, "cluster_id": cluster_id, "prompt_num": args.prompt_num}
                        with open(data_json_path, "a") as json_file:
                            json.dump(data, json_file)
                            json_file.write('\n')

                print("Finished writing processed datapoints part %d to %s"%(file_idx, data_json_path))
                
        print("Finished saving selected instructions for %s!"%(split_json_file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="",
        help="Directory path of the form ./1_mdcure_generation/generations_general/prompt_<prompt_id>, where prompt_id is defined as in generate_general.py"
        ) 
    parser.add_argument(
        "--start_file_idx", 
        type=int, 
        default=0,
        ) 
    parser.add_argument(
        "--start_split_idx", 
        type=int, 
        default=0,
        ) # 0=train, 1=val, 2=test
    parser.add_argument(
        "--prompt_num",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    main(args)
