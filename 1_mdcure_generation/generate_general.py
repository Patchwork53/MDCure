import torch
import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from ast import literal_eval
import pandas as pd 
from openai import OpenAI

# set up OpenAI client
client = OpenAI()

def get_context(text, target_sentence):
    sentences = text.split('\n')
    
    # find the index of the target sentence
    target_index = sentences.index(target_sentence)
    
    # extract context sentences
    start_index = max(0, target_index - 3)
    end_index = min(len(sentences), target_index + 4)
    context = sentences[start_index:end_index]
    
    return " ".join(context)

def gpt_generate(prompt, model):
    return client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                n=1,
                temperature=1,
            )

def promptA(snippet1, snippet2, context1, context2): 
    prompt = f"""Snippets: '{snippet1}', '{snippet2}'\nContext Paragraphs: '{context1}', '{context2}'\nBased on the given snippets and context paragraphs, construct an instruction-answer pair such that (1) the answer is based on the two snippets and (2) the instruction is a plausible prompt or question to which the answer would be the expected response. Make sure both snippets are required to answer the instruction. You will be penalized if the instruction concerns only one snippet. Format your response as:\nInstruction: <prompt or question>\nAnswer: <answer>"""
    return prompt

def promptB(snippet1, snippet2, context1, context2):
    prompt = f"""Snippets: '{snippet1}', '{snippet2}'\n"""
    direction = """Based on the given snippets, construct an instruction-answer pair such that (1) the answer is yes and (2) the instruction is a plausible prompt or question to which yes would be the expected response. Make sure the answer does not conflict with the information in the snippets. You will be penalized if the instruction-answer pair is unfactual. Do NOT mention the snippets in the instruction. Format your response as:\nInstruction: <prompt or question>\nAnswer: <yes>"""
    return prompt+direction

def promptC(snippet1, snippet2, context1, context2):
    prompt = f"""Snippets: '{snippet1}', '{snippet2}'\n"""
    direction = """Based on the given snippets, construct an instruction-answer pair such that (1) the answer is no and (2) the instruction is a plausible prompt or question to which no would be the expected response. Make sure the answer does not conflict with the information in the snippets. You will be penalized if the instruction-answer pair is unfactual. Do NOT mention the snippets in the instruction. Format your response as:\nInstruction: <prompt or question>\nAnswer: <no>"""
    return prompt+direction

def promptD(snippet1, snippet2, context1, context2):
    prompt = f"""Snippets: '{snippet1}', '{snippet2}'\nContext Paragraphs: '{context1}', '{context2}'\n"""
    direction = """Based on the given snippets and context paragraphs, construct an instruction-answer pair such that (1) the answer is a brief phrase and NOT a sentence and (2) the instruction is a plausible prompt or question to which the answer is the expected response. Make sure both snippets are required to answer the instruction. You will be penalized if the instruction concerns only one snippet. Make sure the answer is a brief phrase less than 7 words in length, with NO periods. You will be penalized if the answer is longer than 7 words or if the answer is a sentence. Format your response as:\nInstruction: <prompt or question>\nAnswer: <answer>"""
    return prompt+direction

def promptE(snippet1, snippet2, context1, context2):
    prompt = f"""Context Paragraphs: '{context1}', '{context2}'\n"""
    direction = """Based on the two given context paragraphs, construct an instruction-answer pair such that (1) the answer is summary of the two paragraphs and (2) the instruction is a plausible prompt or question to which the answer is the expected response. Make sure both paragraphs are required to answer the instruction. You will be penalized if the instruction concerns only one paragraph. Make sure the answer does not conflict with the information in the paragraphs. You will be penalized if the instruction-answer pair is unfactual. Make sure the answer is at least 5 sentences in length. Do not mention the context paragraphs in the instruction. Format your response as:\nInstruction: <prompt or question>\nAnswer: <answer>"""
    return prompt+direction

def promptF(snippet1, snippet2, context1, context2):
    prompt = f"""Context Paragraphs: '{context1}', '{context2}'\n"""
    direction = """Based on the two given context paragraphs, construct an instruction-answer pair such that (1) the answer is summary of the two paragraphs and (2) the instruction is a plausible prompt or question to which the answer is the expected response. Make sure both paragraphs are required to answer the instruction. You will be penalized if the instruction concerns only one paragraph. Make sure the answer does not conflict with the information in the paragraphs. You will be penalized if the instruction-answer pair is unfactual. Make sure the answer is less than 5 sentences in length. Do not mention the context paragraphs in the instruction. Format your response as:\nInstruction: <prompt or question>\nAnswer: <answer>"""
    return prompt+direction

# Function to generate and save instruction tuning prompts/examples
def generate_and_save_instructions(
    input_dir,
    output_dir,
    prompt_num,
    num_prompts_to_generate,
    model,
    use_existing_prompts,
    given_prompt_dir,
    start_instr_file_idx,
    num_slices_to_do,
    only_generate_prompts,
    use_train,
    use_valid,
):
    # ensure output dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # access source data
    if use_train:
        all_files = [
            f
            for f in os.listdir(input_dir)
            if f.endswith("train.pt")
        ]
        num_rows_to_get_overall = args.num_rows_to_get_overall
    elif use_valid:
        all_files = [
            f
            for f in os.listdir(input_dir)
            if f.endswith("valid.pt")
        ]
        num_rows_to_get_overall = args.num_rows_to_get_overall*0.1
    else: 
        all_files = [
            f
            for f in os.listdir(input_dir)
            if f.endswith(".pt")
        ]
    all_files = all_files[::-1]
    count = 0

    # consider each data split (train/valid/test)
    for file_idx, file_name in enumerate(tqdm(all_files)):
        print(file_name)
        num_prompts=0

        all_prompts = []
        instr_cluster_ids = []
        source_docs = []
        num_snips = []
        answers = []
        prompt_dir = os.path.join(output_dir, "prompts")
        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir)
        docs_dir = os.path.join(output_dir, "source_docs")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
        cluster_id_dir = os.path.join(output_dir, "cluster_ids")
        if not os.path.exists(cluster_id_dir):
            os.makedirs(cluster_id_dir)
    
        if not use_existing_prompts:
            base_snippet_pairs_path = "./0_source_data_preparation/base_snippet_pairs/" + file_name.replace('.pt', '.csv')
            sent_pairs_all = pd.read_csv(base_snippet_pairs_path, index_col=None, converters={"articles": literal_eval,"answer": literal_eval, "cluster_id": literal_eval})

            finished_pairs = sent_pairs_all.sample(n=10000, random_state=42)
            finished_pairs['answer'] = finished_pairs['answer'].apply(tuple)
            finished_pairs['articles'] = finished_pairs['articles'].apply(tuple)
            sent_pairs_all['answer'] = sent_pairs_all['answer'].apply(tuple)
            sent_pairs_all['articles'] = sent_pairs_all['articles'].apply(tuple)
            
            merged = pd.merge(sent_pairs_all, finished_pairs, how='left', indicator=True)
            unfinished_pairs = merged[merged['_merge'] == 'left_only']

            # select indicated # of samples
            selected_sent_pairs = unfinished_pairs.sample(n=num_rows_to_get_overall, random_state=42) 
            selected_sent_pairs = selected_sent_pairs.iloc[(prompt_num-1)*num_prompts_to_generate:prompt_num*num_prompts_to_generate]
            selected_sent_pairs['answer'] = selected_sent_pairs['answer'].apply(tuple)
            selected_sent_pairs['articles'] = selected_sent_pairs['articles'].apply(tuple)
            selected_sent_pairs.drop('_merge', axis=1, inplace=True)
            
            print("PROMPT NUM", prompt_num, "DIMENSIONS", (prompt_num-1)*num_prompts_to_generate, prompt_num*num_prompts_to_generate)

            if prompt_num in {3,4}:
                selected_sent_pairs = selected_sent_pairs.iloc[:len(selected_sent_pairs)//2]

            all_source_docs = selected_sent_pairs['articles'].tolist()
            all_answers = selected_sent_pairs['answer'].tolist()
            all_cluster_ids = selected_sent_pairs['cluster_id'].tolist()
            file_idx = 0

            for docs, snippets, cluster_id in zip(all_source_docs, all_answers, all_cluster_ids):
                context0 = get_context(docs[0], snippets[0])
                context1 = get_context(docs[1], snippets[1])

                if prompt_num==2:
                    prompt = promptA(snippets[0], snippets[1], context0, context1)
                elif prompt_num==3:
                    prompt = promptB(snippets[0], snippets[1], context0, context1)
                elif prompt_num==4:
                    prompt = promptC(snippets[0], snippets[1], context0, context1)
                elif prompt_num==5:
                    prompt = promptD(snippets[0], snippets[1], context0, context1)
                elif prompt_num==6:
                    prompt = promptE(snippets[0], snippets[1], context0, context1)
                elif prompt_num==7:
                    prompt = promptF(snippets[0], snippets[1], context0, context1)

                source_docs.append([docs[0], docs[1]])
                all_prompts.append(prompt)
                instr_cluster_ids.append(cluster_id)
                num_prompts += 1

                if num_prompts > 9:
                    prompt_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                    torch.save(all_prompts, os.path.join(prompt_dir, prompt_file_name))
                    print("Finished saving prompt slice %d at %s"%(file_idx, prompt_dir+"/"+prompt_file_name))

                    cluster_id_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                    torch.save(instr_cluster_ids, os.path.join(cluster_id_dir, cluster_id_file_name))
                    print("Finished saving cluster ids slice %d at %s"%(file_idx, cluster_id_dir+"/"+cluster_id_file_name))

                    source_docs_file_name = file_name[:-2]+"%d.pt"%(file_idx)
                    torch.save(source_docs, os.path.join(docs_dir, source_docs_file_name))
                    print("Finished saving source documents slice %d at %s"%(file_idx, docs_dir+"/"+source_docs_file_name))

                    print("Just-saved number of prompts:",num_prompts)
                    num_prompts = 0
                    source_docs = []
                    all_prompts = []
                    instr_cluster_ids = []
                    file_idx += 1
            
            torch.save(all_prompts, os.path.join(prompt_dir, file_name))
            print("Finished saving last few prompts at %s"%(prompt_dir+"/"+file_name))

            torch.save(instr_cluster_ids, os.path.join(cluster_id_dir, file_name))
            print("Finished saving last few cluster_ids at %s"%(cluster_id_dir+"/"+file_name))

            torch.save(source_docs, os.path.join(docs_dir, file_name))
            print("Finished saving last few source documents at %s"%(docs_dir+"/"+file_name))

            print("Just-saved number of prompts:", num_prompts)
        
    ######################################################
    ######## Use prompts to generate instructions ########
    ######################################################
    if only_generate_prompts==False:
        if use_existing_prompts:
            prompt_dir = given_prompt_dir

        # ensure output directory exists
        instr_dir = os.path.join(output_dir, "instructions")
        if not os.path.exists(instr_dir):
            os.makedirs(instr_dir)
    
        all_files = sorted([
            f
            for f in os.listdir(prompt_dir)
            if f.endswith(".pt")
        ])

        # consider each split (train/valid/test)
        num_slices_done = 0
        for file_idx, pt_file in enumerate(tqdm(all_files)):

            if num_slices_to_do!= -1 and num_slices_done > num_slices_to_do:
                break
                                
            if start_instr_file_idx!=-1 and file_idx < start_instr_file_idx:
                continue
            
            elif (start_instr_file_idx==-1) or (start_instr_file_idx!=-1 and file_idx >= start_instr_file_idx): 

                # write current filename to log (to assist pause/resume)
                print("#"*20)
                print("On slice", file_idx)
                f = open(os.path.join(output_dir,"num_instr_slices_generated_so_far.txt"), "w")
                f.write("\nCurerntly generating instructions for prompt file: "+str(pt_file))
                f.write("\nCorresponding file_idx: "+str(file_idx))
                f.close()

                # prompt GPT
                prompt_slice = torch.load(os.path.join(prompt_dir, pt_file))
                instructions = [gpt_generate(prompt, model).choices[0].message.content.strip() for prompt in prompt_slice]

                instr_file_name = pt_file

                torch.save(instructions, os.path.join(instr_dir, instr_file_name))
                print("Finished saving generated instructions part %d at %s"%(file_idx, instr_dir+"/"+instr_file_name))

                num_slices_done += 1

        print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./0_source_data_preparation/base_articles",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./1_mdcure_generation/generations_general",
    )
    parser.add_argument(
        '--use_existing_prompts', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--only_generate_prompts', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--given_prompt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_instr_file_idx",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--num_prompts_to_generate",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--prompt_num",
        type=int,
        default=1, # prompt template A = prompt_num 2, prompt template B = prompt_num 3, etc.
    )
    parser.add_argument(
        "--num_slices_to_do",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        '--use_train', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--use_valid', 
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--num_rows_to_get_overall', 
        type=int,
        default=300000,
    )
    args = parser.parse_args()
    print(args)

    
    if args.use_existing_prompts:
        output_dir = args.given_prompt_dir.replace("/prompts", "")
    else:
        output_dir = f"{args.output_dir}/prompt_{args.prompt_num}"

    generate_and_save_instructions(
        args.input_dir,
        output_dir,
        args.prompt_num,
        args.num_prompts_to_generate,
        args.model,
        args.use_existing_prompts,
        args.given_prompt_dir,
        args.start_instr_file_idx,
        args.num_slices_to_do,
        args.only_generate_prompts,
        args.use_train,
        args.use_valid,
    )

