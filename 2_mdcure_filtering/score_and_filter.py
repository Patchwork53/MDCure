import argparse
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, LlamaConfig, PreTrainedModel, LlamaForSequenceClassification
import json
import numpy as np
import torch.nn as nn

WEIGHT = torch.tensor([1/9, 1/9, 2/9, 2/9, 2/9, 2/9], device="cuda") # weights to apply across the 6 scoring criteria for weighted average

# Login to HF to access LLAMA model
from huggingface_hub import login
login("")

class RewardModelConfig(LlamaConfig):
    model_type = "RewardModel"

    def __init__(self, reward_dim=None, base_model_name=None, **kwargs):
        super().__init__(**kwargs)
        
        self.reward_dim = reward_dim
        self.base_model_name = base_model_name

class RewardModel(PreTrainedModel):
    config_class = RewardModelConfig

    def create_base_model(self):
        
        # use sequence classification model for consistency with https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1 
        BACKBONE_MODEL =  LlamaForSequenceClassification.from_pretrained( 
            self.config.base_model_name,
            config=LlamaConfig.from_pretrained(self.config.base_model_name),
        )
        BACKBONE_MODEL.config.pad_token_id = BACKBONE_MODEL.config.eos_token_id
        BACKBONE_MODEL.config.output_hidden_states = True

        for param in BACKBONE_MODEL.parameters():
            param.requires_grad = False

        return BACKBONE_MODEL

    def __init__(self, config):
        super(RewardModel, self).__init__(config)
        
        # use .base_model to remove lm_head
        self.BASE_MODEL = self.create_base_model().base_model 

        # regression head for reward prediction
        self.regression_head = nn.Linear(self.BASE_MODEL.config.hidden_size, config.reward_dim)
        
    def forward(self, input_ids, attention_mask=None, rewards=None, **kwargs):

        # forward pass through the base model
        outputs = self.BASE_MODEL(input_ids, attention_mask=attention_mask, **kwargs)
        
        hidden_states = outputs.hidden_states[-1]

        # access hidden state corresponding to the last token in each sequence across the batch
        last_token_hidden_state = hidden_states[:, -1, :] 
        reward_predictions = self.regression_head(last_token_hidden_state)

        return reward_predictions

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.BASE_MODEL.prepare_inputs_for_generation(*args, **kwargs)
    
def main(args):

    # load Files: train/valid_with_length_direction
    files = [
        os.path.join(args.input_dir, "train_with_length_direction.json"),
        os.path.join(args.input_dir, "valid_with_length_direction.json"),
    ]

    # create output dirs
    if args.score_general:
        output_dir = f"./2_mdcure_filtering/scored_instructions/general"
    elif args.score_style_specific:
        output_dir = f"./2_mdcure_filtering/scored_instructions/style_specific"
    os.makedirs(output_dir, exist_ok=True)

    # load scoring model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.scoring_model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    AutoConfig.register("RewardModel", RewardModelConfig)
    AutoModel.register(RewardModelConfig, RewardModel)

    rm = AutoModel.from_pretrained(args.scoring_model_path)
    print(rm)

    if torch.cuda.is_available():
        rm = rm.to(torch.device("cuda"))

    # iterate over train/val splits
    for file in files:
        print("##################")
        print(f"Working on {file}")

        # read instruction data file as json
        data_df = pd.read_json(file, lines=True)

        # score instruction data
        for index, row in data_df.iterrows():
            print(f"On index {index}")

            # get instruction & context
            instruction_input = row['instruction']
            instruction = instruction_input.split("\n\n")[-1]
            context = instruction_input.removesuffix(instruction).strip()
            num_docs = len(context.split("\n\n"))
                    
            # obtain scores per criterion
            input_text = f"Instruction: {instruction}\n\n{context}"
            tokenized_input = tokenizer(
                                    input_text, 
                                    return_tensors='pt', 
                                    truncation=True, 
                                    padding=True,
                                    ).to(torch.device("cuda"))
            all_six_scores = rm(tokenized_input["input_ids"]).squeeze(0) # flatten for dot product
            all_six_scores = all_six_scores*4. + 1. # scale up to 1->5 range

            score = torch.dot(all_six_scores, WEIGHT).cpu().item()

            # create output dirs to bin by threshold
            thresh3_dir = f"{output_dir}/thresh3"
            thresh35_dir = f"{output_dir}/thresh35"
            thresh4_dir = f"{output_dir}/thresh4"
            os.makedirs(thresh3_dir, exist_ok=True)
            os.makedirs(thresh35_dir, exist_ok=True)
            os.makedirs(thresh4_dir, exist_ok=True)

            # create output paths
            if "train" in file:
                thresh3_path = f"{thresh3_dir}/train_with_LD.json"
                thresh35_path = f"{thresh35_dir}/train_with_LD.json"
                thresh4_path = f"{thresh4_dir}/train_with_LD.json"
            elif "valid" in file:
                thresh3_path = f"{thresh3_dir}/valid_with_LD.json"
                thresh35_path = f"{thresh35_dir}/valid_with_LD.json"
                thresh4_path = f"{thresh4_dir}/valid_with_LD.json"

            # create final data instance
            if args.score_general:
                datum = {
                    "instruction": row['instruction'], 
                    "answer": row['answer'], 
                    "cluster_id": row['cluster_id'], 
                    "prompt_num": args.prompt_num,
                    "num_docs": num_docs,
                    "score": score, 
                    "six_scores": str(all_six_scores.cpu().detach().numpy().tolist()),
                    }
            elif args.score_style_specific:
                datum = {
                    "instruction": row['instruction'], 
                    "answer": row['answer'], 
                    "cluster_id": row['cluster_id'], 
                    "prompt_num": row['prompt_num'],
                    "prompt_id": row['prompt_id'], 
                    "num_docs": row['num_docs'],
                    "score": score,
                    "six_scores": str(all_six_scores.cpu().detach().numpy().tolist()),
                    "slice": args.slice_idx,
                }
            
            # save high-scoring instructions
            if score >= 3:
                with open(thresh3_path, "a") as json_file:
                    json.dump(datum, json_file)
                    json_file.write('\n')

                if score >= 3.5:
                    with open(thresh35_path, "a") as json_file:
                        json.dump(datum, json_file)
                        json_file.write('\n')
                
                    if score >= 4:
                        with open(thresh4_path, "a") as json_file:
                            json.dump(datum, json_file)
                            json_file.write('\n')

        print(f"Finished saving selected instructions for {file}!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir", 
        type=str,
    ) 
    parser.add_argument(
        "--scoring_model_path",
        type=str,
    )
    parser.add_argument(
        "--prompt_num",
        type=int, 
        default=0, # manually provide if desired for the general-template-generated samples
    )
    parser.add_argument(
        "--score_general", 
        action="store_true",
        default=False, 
    ) 
    parser.add_argument(
        "--score_style_specific", 
        action="store_true",
        default=False, 
    ) 
    args = parser.parse_args()

    main(args)
