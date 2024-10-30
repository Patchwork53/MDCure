import argparse
from tqdm import tqdm
import pandas as pd
import re
import random
import os
import torch
import json
from nltk.tokenize import sent_tokenize

def get_query(instr: str):
    """
    Helper function to extract input and output from the generated instruction
    """
    if "Exam Question" in instr:
        break_idx = instr.index("Answer:")
        query_with_header = instr[:break_idx].strip()
        answer_with_header = instr[break_idx:].strip()
        query = query_with_header.removeprefix("Exam Question:")
        answer = answer_with_header.removeprefix("Answer:")

    else: 
        try:
            break_idx = instr.index("**Answer:**")
            query_with_header = instr[:break_idx].strip()
            answer_with_header = instr[break_idx:].strip()
            answer = answer_with_header.removeprefix("**Answer:**")
            try:
                query = query_with_header.removeprefix("**Question:**")
            except:
                query = query_with_header.removeprefix("Question:")
        except:
            try:
                break_idx = instr.index("Answer:")
                query_with_header = instr[:break_idx].strip()
                answer_with_header = instr[break_idx:].strip()
                query = query_with_header.removeprefix("Question:")
                answer = answer_with_header.removeprefix("Answer:")
            except: 
                try: 
                    break_idx = instr.index("A: ")
                    query_with_header = instr[:break_idx].strip()
                    answer_with_header = instr[break_idx:].strip()
                    query = query_with_header.removeprefix("Q:")
                    answer = answer_with_header.removeprefix("A:")
                except: 
                    query = ""
                    answer = ""

    # Edge case: if more than 1 question/answer was generated:
    if "Question:" in query or "Answer:" in query or "Question:" in answer or "Answer:" in answer:
        query = ""
        answer = ""

    return query.strip(), answer.strip()

def get_length_info(finalized_answer: str):

    # count number of sentences in answer
    num_sents = len(sent_tokenize(finalized_answer))
    length_spec = ""

    # count number of words in answer
    if num_sents == 1:
        num_words = len(finalized_answer.split())
    
    if num_sents==1:
        if num_words<3:
            length_spec += "1-2 words"
        elif num_words<5:
            length_spec += "3-4 words"
        elif num_words<7:
            length_spec += "a phrase of at least 5-6 words"
        else:
            length_spec += "1-2 sentences"
    elif num_sents==2:
        length_spec += "1-2 sentences"
    elif num_sents in [3,4]:
        length_spec += "3-4 sentences"
    elif num_sents in [5,6,7]:
        length_spec += "5-7 sentences"
    elif num_sents in [8,9,10]:
        length_spec += "8-10 sentences"
    else:
        length_spec += "at least 10 sentences"
    length_direction = random.choice(length_templates).format(length_spec=length_spec)

    return length_spec, length_direction

def finalize_instr(
                  prompt: str,
                  question: str, # pre-stripped when passed in!
                  answer: str,
                  cluster: list, # list of source docs
                  prompt_num=None,
                  ):

    # obtain documents & answers
    if prompt_num!=-1:
        if prompt_num==1:
            context_docs = prompt.strip("A question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed is:\nQuestion: <respond here>\nAnswer: <respond here briefly>")

        elif prompt_num==2:
            context_docs = prompt.strip("What is a question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\nQuestion: <respond here>\nAnswer: <respond here briefly>")

        elif prompt_num==3:
            prompt = prompt.strip("Articles:")
            context_docs = prompt.strip("What is an exam question that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\n\nExam Question: <respond here>\nAnswer: <respond here briefly>").strip()

        elif prompt_num==4:
            context_docs = prompt.strip("What is a question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\nQuestion: <respond here>\nAnswer: <respond here, feel free to use a single word or phrase>")

        elif prompt_num==5:
            context_docs = prompt.strip("A question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed is:\nQuestion: <respond here>\nAnswer: <respond here>")

        elif prompt_num==6:
            context_docs = prompt.strip("What is a question or command that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\nQuestion: <respond here>\nAnswer: <respond here, using ONLY a single word or phrase>")

        elif prompt_num==7:
            prompt = prompt.strip("Articles:")
            context_docs = prompt.strip("Contrasting Question: <respond here>\nAnswer: <respond here briefly>")

        elif prompt_num==8:
            prompt = prompt.strip("Articles:")
            context_docs = prompt.strip("What is an exam question that can ONLY be answered by utilizing ALL of the above documents and that CANNOT be answered if any one document is removed?\n\nExam Question: <respond here>\nAnswer Choices: <respond here>\nAnswer: <answer letter only>")

    else: # if prompt type is just template
        break_idx = prompt.index("### Articles:")
        end_idx = prompt.index("Question: <respond here>")
        sandwiched_docs = prompt[break_idx:end_idx].strip("### Articles:").strip()
        context_docs = sandwiched_docs.split("\n\n")

    # if the input in the final instruction should utilize ALL cluster docs
    if args.use_all_cluster_docs: 
        temp = [doc.replace("\n", " ") for doc in cluster]
        docs_str = "\n\n".join(temp)
    
    # otherwise, if the input in final instrxn should only use relevant docs
    else:
        docs_str = context_docs

    finalized_instr = f"{docs_str}\n\n{question}"
    formatted_finalized_instr = f"Documents:\n{docs_str}\n\nQuestion: {question}"

    return finalized_instr, answer, context_docs, formatted_finalized_instr

answer_lengths = [
    "1-2 words",
    "3-4 words",
    "a phrase of at least 5-6 words",
    "1-2 sentences",
    "3-4 sentences",
    "6 sentences",
    "8 sentences",
    "10 sentences",
]

length_templates = [
    """Answer with {length_spec}.""",
    """Answer using {length_spec}.""",
    """Respond with {length_spec}.""",
    """Respond using {length_spec}.""",
    """Formulate your answer in {length_spec}.""",
    """Reply with a {length_spec} answer.""",
    """Craft your response in {length_spec}.""",
    """Give a response that is {length_spec}.""",
    """Answer in around {length_spec}.""",
]

def main(args):
    
    # determine what type of prompt was used to obtain these instructions
    input_dir = args.input_dir.removeprefix("./generations_style_specific/").strip("/")
    prompt_type = ""

    # if one of templates G-N was used
    if "prompt" in input_dir:
        prompt_type = "straightforward"
        prompt_id, num_docs_in_context = input_dir.split("_numdocs_")
        num_docs_in_context = int(num_docs_in_context)
        prompt_num = int(prompt_id[-1])

    # if style-specific template was used
    elif "template" in input_dir:
        prompt_type = "template"
        prompt_id, num_docs_in_context = input_dir.split("_numdocs_")
        num_docs_in_context = int(num_docs_in_context)
        length_idx = int(prompt_id[-1]) 
        length_spec = answer_lengths[length_idx]
        length_direction = random.choice(length_templates).format(length_spec=length_spec)
        prompt_num=-1

    # ensure save directory exists
    raw_output_dir = os.path.join(args.input_dir, "data_jsons")
    os.makedirs(raw_output_dir, exist_ok=True)

    # get paths to input info
    prompt_dir = os.path.join(args.input_dir, "prompts")
    instr_dir = os.path.join(args.input_dir, "instructions")
    cluster_ids_dir = os.path.join(args.input_dir, "cluster_ids") 

    # sort alphanumerically so that names match in order of reference 
    all_prompt_files_train = sorted(sorted([
                f
                for f in os.listdir(prompt_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_prompt_files = [all_prompt_files_train]

    all_instr_files_train = sorted(sorted([
                f
                for f in os.listdir(instr_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_instr_files = [all_instr_files_train]

    all_cluster_id_files_train = sorted(sorted([
                f
                for f in os.listdir(cluster_ids_dir)
                if f.endswith(".pt") and "train" in f
            ], key=str.lower), key=len)
    all_cluster_id_files = [all_cluster_id_files_train]

    # iterate over train/val/test splits
    for split_idx, (prompt_split_files, instr_split_files, cluster_id_files) in enumerate(tqdm(zip(all_prompt_files, all_instr_files, all_cluster_id_files))):

        if instr_split_files==[] or split_idx<args.start_split_idx:
            continue

        # create json file to save data for current split
        one_prompt_pt = instr_split_files[0] # e.g., train.1.pt 
        print("FILENAMES:",prompt_split_files, instr_split_files, cluster_id_files)
        split_json_file_name = one_prompt_pt[:one_prompt_pt.index(".")] + ".json" # e.g., train.1.pt -> train.json
        raw_data_json_path = os.path.join(raw_output_dir, split_json_file_name)

        print("########################")
        print("Working on",split_json_file_name)
        
        # score all the instructons for current split
        for file_idx, instr_pt in enumerate(tqdm(instr_split_files)):
            
            # write current filename to log (in case need to pause/resume)
            f = open(os.path.join(raw_output_dir, "num_slices_processed_so_far.txt"), "w")
            f.write("\nOn instruction/prompt file: "+str(instr_pt))
            f.write("\nCorresponding file_idx: "+str(file_idx))
            f.close()
            
            # iterate through the data .pt files; 
            # Note: Use len(N), N = desired # of files to process
            if args.start_file_idx <= file_idx < len(instr_split_files):

                # access current slice
                prompt_slice = torch.load(os.path.join(prompt_dir, instr_pt))
                instr_slice = torch.load(os.path.join(instr_dir, instr_pt))
                cluster_id_slice = torch.load(os.path.join(cluster_ids_dir, instr_pt))

                # access source doc clusters for curernt slice
                doc_dir = os.path.join(args.input_dir,"source_docs")
                source_doc_slice = torch.load(os.path.join(doc_dir, instr_pt))

                # extract & store each example in the current slice
                for example_idx, (prompt, instr, cluster_id, cluster) in enumerate(zip(prompt_slice, instr_slice, cluster_id_slice, source_doc_slice)):
                    
                    # get the question and answer from the GPT-generated response (instr)
                    question, answer = get_query(instr.strip())

                    # if the question and answer are valid
                    if question != "" and answer != "":

                        # obtain the normally-formatted finalized instruction and answer
                        # Note: finalized_instr = context docs + question
                        finalized_instr, finalized_answer, context_docs, formatted_finalized_instr = finalize_instr(prompt, question, answer, cluster, prompt_num) 

                        # obtain length info if a template G-N was used
                        if prompt_type == "straightforward":
                            if prompt_num not in [3, 8]:
                                length_spec, length_direction = get_length_info(finalized_answer)
                            else: 
                                length_spec, length_direction = "", ""
                                
                        # attach length direction to final instruction
                        finalized_instr_with_length_direction = f"{finalized_instr} {length_direction}".strip()

                        # save the datum
                        data = {
                            "gpt_raw_output": instr,
                            "context_docs_used_for_gpt_generation": context_docs,
                            "all_cluster_docs": cluster,
                            "instruction": finalized_instr_with_length_direction,
                            "answer": finalized_answer, 
                            "cluster_id": cluster_id,
                            "prompt_num": prompt_num, # -1 for template-based
                            "prompt_id": prompt_id, # indicate style-specific vs general
                            "length_spec": length_spec, 
                            "length_direction": length_direction,
                            "format direction": "",
                        }
                        with open(raw_data_json_path, "a") as json_file:
                            json.dump(data, json_file)
                            json_file.write('\n')

                print("Finished writing processed datapoints part %d to %s"%(file_idx, raw_data_json_path)) 

        print("Finished saving selected instructions for %s!"%(split_json_file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="",
        help="Directory path of the form ./1_mdcure_generation/generations_style_specific/..., where prompt_id is defined as in generate_style_specific.py"
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
        '--use_all_cluster_docs', 
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    main(args)
