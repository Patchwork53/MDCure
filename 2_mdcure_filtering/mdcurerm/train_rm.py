import os
import json
import wandb
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datasets import Dataset, DatasetDict, load_from_disk

from transformers import AutoTokenizer, Trainer, PreTrainedModel, AutoModel, LlamaConfig, AutoConfig, LlamaForSequenceClassification
from transformers import TrainingArguments, HfArgumentParser
from transformers import DataCollatorWithPadding
from trl import ModelConfig

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

        # hidden state corresponding to the last token in each sequence across the batch
        last_token_hidden_state = hidden_states[:, -1, :] 
        reward_predictions = self.regression_head(last_token_hidden_state)

        return reward_predictions

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.BASE_MODEL.prepare_inputs_for_generation(*args, **kwargs)

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
 
        rewards = inputs['rewards']
        reward_predictions = outputs
        
        loss_fct = nn.MSELoss(reduction="mean")
        loss = loss_fct(reward_predictions, rewards)
        
        return (loss, outputs) if return_outputs else loss

class RewardDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        rewards = torch.stack([torch.tensor(f['rewards'], dtype=torch.float32) for f in features])
        batch['rewards'] = rewards
        return batch

def create_dataset_splits(ds, test_size=0.075, valid_size=0.075, output_dir="./2_mdcure_filtering/splits"):
    print(f"Creating new split data and saving to {output_dir}")
    dataset = ds.train_test_split(test_size=test_size)
    train_valid_split = dataset['train'].train_test_split(test_size=valid_size / (1-test_size))
    dataset = DatasetDict({
        'train': train_valid_split['train'],
        'valid': train_valid_split['test'],
        'test': dataset['test']
    })
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    
    return dataset

def preprocess_function(examples):
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "rewards": [],
    }

    for context, instruction, r, cf, cr, ci, ir, co in \
        zip(examples["context"],
            examples["instruction"], 
            examples["Relevance"],
            examples["Coherence & Factuality"],
            examples["Creativity"],
            examples["Context Integration"],
            examples["Inter-Document Relationships"],
            examples["Complexity"]):
        
        input_text = instruction + "\n\n" + context
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)

        rewards = [(float(x) - 1.) / 4. for x in [r, cf, cr, ci, ir, co]] # normalize to range 0-1 for consistency with paper; technically not needed since all ratings are along the same scale

        new_examples["input_ids"].append(inputs["input_ids"].squeeze(0))
        new_examples["attention_mask"].append(inputs["attention_mask"].squeeze(0))
        new_examples["rewards"].append(rewards)

    return new_examples

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    preds = torch.tensor(predictions).squeeze()
    labels = torch.tensor(labels).squeeze()
    mse = torch.nn.functional.mse_loss(preds, labels, reduction='mean').item()
    result = {"loss": mse}

    return result

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

    # Parse HF args
    hf_parser = HfArgumentParser((TrainingArguments, ModelConfig))
    training_args, model_config, remaining_args = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if remaining_args == []:
        raise Exception("Must pass in non-HF arguments!")

    # Parse non-HF args
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_run_name')
    parser.add_argument('--preference_data_path', default="2_mdcure_filtering/preference_data/parsed_ratings.jsonl")
    parser.add_argument('--base_model_name', default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    args = parser.parse_args(remaining_args)
    
    # Create output dir
    output_dir = os.path.join(training_args.output_dir, args.wandb_run_name)
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = output_dir # format: ./results/{wandb_run_name}

    # Set up wandb logging
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb.login()
        wandb.init(
            project="reward-model-training", 
            config={
                "model_name_or_path": args.base_model_name,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_train_epochs": training_args.num_train_epochs
            },
            name=args.wandb_run_name,
        )

    ###################
    # Model & Tokenizer
    ###################
    AutoConfig.register("RewardModel", RewardModelConfig)
    AutoModel.register(RewardModelConfig, RewardModel)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    reward_config = RewardModelConfig(base_model_name=args.base_model_name, reward_dim=6) 
    reward_model = RewardModel(reward_config)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')
    
    ###################
    # Dataset
    ###################
    random.seed(42)

    file_path = args.preference_data_path
    split_path = os.path.join(output_dir, "splits")

    if os.path.exists(split_path):
        print(f"Loading existing split data from {split_path}")
        dataset_splits = load_from_disk(split_path)

    else:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        random.shuffle(data)
        raw_dataset = Dataset.from_list(data)
        dataset_splits = create_dataset_splits(raw_dataset, output_dir=split_path)

    dataset_splits = dataset_splits.map(
        preprocess_function,
        batched=True,
    )
    dataset_splits.set_format(type='torch', columns=['input_ids', 'attention_mask', 'rewards'])

    print("NUM_TRAIN_SAMPLES:", len(dataset_splits['train']))
    print("NUM_VALID_SAMPLES:", len(dataset_splits['valid']))
    print("NUM_TEST_SAMPLES:", len(dataset_splits['test']))

    ###################
    # Trainer Setup
    ###################
    collator = RewardDataCollator(tokenizer)

    trainer = RewardTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_splits['train'],
        eval_dataset=dataset_splits['valid'],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb.config.update({
            "train_samples": len(dataset_splits['train']), 
            "valid_samples": len(dataset_splits['valid']), 
            "test_samples": len(dataset_splits['test'])
        })

    ###################
    # Training
    ###################
    final_save_dir = os.path.join(output_dir, "final_model")
    trainer.train()
    trainer.save_model(final_save_dir)

    ###################
    # Test Evaluation
    ###################
    test_metrics = trainer.evaluate(
        dataset_splits['test'], 
        metric_key_prefix="test",
        )
    print("Evaluation Results on Test Dataset:")
    print(test_metrics)

    # Visualize predictions 
    preds = trainer.predict(
        dataset_splits['test'], 
        metric_key_prefix="test",
        )
    print(preds.predictions)
    df = pd.DataFrame(preds.predictions)
    df.to_csv(os.path.join(output_dir, 'final_test_preds.csv'), index=False, header=False)
    
    