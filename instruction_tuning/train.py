import wandb
import argparse
import numpy as np
import os
import torch
import transformers
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, AutoModel, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from evaluate import load
import json
import re, string
import pandas as pd
from datasets import Dataset, DatasetDict
import random
from transformers import set_seed as hf_set_seed
from peft import PeftModel

def main(args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    hf_set_seed(args.random_seed)

    if args.use_unsloth:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from unsloth import is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template

    ### Set up wandb logging
    if args.use_peft or args.no_gpu or (not args.use_peft and int(os.environ["LOCAL_RANK"])==0):
        wandb.login()
        if args.wandb_run_id:
            run = wandb.init(
                project=args.wandb_project_name,
                group=args.wandb_group,
                resume="must", 
                id=args.wandb_run_id,
                )   
        else:
            run = wandb.init(
                project=args.wandb_project_name,
                group=args.wandb_group,
                ) 
        
    model_name_str = args.short_model_name if args.short_model_name else args.model_name.replace("/", "_").replace("-", "_")
    output_dir = os.path.join(args.output_dir, model_name_str, f"wandb_group_{args.wandb_group}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Checkpoint dir 
    checkpoint_dir = os.path.join(output_dir,"checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    def truncate_multi_doc(
            text: str, 
            tokenizer,
            max_length=4096, 
    ) -> str:
        """
        Given some `text`, which is assumed to be multiple documents joined by some document separator, truncates each document (using `tokenizer`) so that the length of the concatenation of all documents does not exceed max_length.
        """
        doc_sep_mult_factor = 1
        extra_tokens_for_template = 0

        # Don't truncate if not needed
        text_tokens_length = len(tokenizer.tokenize(text))
        if text_tokens_length <= max_length:
            return text

        if "<doc-sep>" in text:
            doc_sep_token = "<doc-sep>"
            q_sep_token = "<q-sep>"
            
        elif "<d>" in text and "</d>" in text:
            doc_sep_token =  "<d>"
            q_sep_token = "<q-sep>"
            doc_sep_mult_factor = 2

        else: 
            doc_sep_token = "\n\n"
            q_sep_token = "\n\n"
        
        if q_sep_token!="\n\n":
            try:
                docs, question = text.split(q_sep_token)
            except: 
                docs, question = text.split("\n\n")
            docs_list = docs.split(doc_sep_token)
        
        # if no special tokens used to break up text components
        else:
            docs_and_question = text.split(doc_sep_token)
            question = docs_and_question[-1]
            docs_list = docs_and_question[:-1]
        
        # -2 to make room for the special tokens, -(len(docs) - 1) to make room for the doc sep tokens.
        # compute [total context length - question length - # of doc sep tokens - # of q_sep tokens + extra spaces from docs that are too short - # spaces to make for any tokens that are included for causal LM input template-ing ] / number of docs
        temp_max_doc_length = (
            max_length - len(tokenizer.tokenize(question)) - doc_sep_mult_factor*(len(docs_list) - 1) - 1 - extra_tokens_for_template
            ) // len(docs_list)

        extra_space = 0
        num_short_docs = 0
        short_doc_lengths = 0
        for doc in docs_list:
            length_dif = temp_max_doc_length - len(tokenizer.tokenize(doc))
            if length_dif > 0:
                # extra_space += length_dif
                short_doc_lengths += len(tokenizer.tokenize(doc))
                num_short_docs += 1

        max_doc_length = (
            max_length - short_doc_lengths - (len(docs_list) - 1) - 1 
            ) // (len(docs_list) - num_short_docs)

        truncated_docs = []
        for doc in docs_list:
            # Truncate each doc to its maximum allowed length
            truncated_docs.append(
                tokenizer.convert_tokens_to_string(
                    tokenizer.tokenize(
                        doc, 
                        max_length=max_doc_length, 
                        truncation=True,
                        )
                ).strip()
            )
        
        if doc_sep_token=="<d>": # Format appropriately if docs enclosed with <d> </d> in original doc-sep notation
            formatted_docs = ["<d>{}</d>".format(s.strip()) for s in truncated_docs]
            revised_docs_str = "".join(formatted_docs)
        else:
            revised_docs_str = doc_sep_token.join(truncated_docs)
        return f"{revised_docs_str}{q_sep_token}{question}"


    def preprocess_function(examples):
        try:
            inputs = [doc for doc in examples['inputs']]
            outputs = [ans for ans in examples['targets']]
        except:
            try:
                inputs = [doc for doc in examples[args.instruction_column_name]]
                outputs = [ans for ans in examples['answer']]
                if inputs == [None]:
                    raise Exception("Empty 'instruction' field")
            except:
                try:
                    inputs = [doc for doc in examples['document']]
                    outputs = [summary for summary in examples['summary']] 
                    if inputs == [None]:
                        raise Exception("Empty 'document' field")  
                except: 
                    try: 
                        inputs = [doc for doc in examples['documents']]
                        outputs = [summary for summary in examples['summary']]   
                        if inputs == [None]:
                            raise Exception("Empty 'documents' field")
                    except: 
                        messages = examples['messages']
                        inputs = [messages[0][0]["content"]]
                        outputs = [messages[0][1]["content"]]
                        if inputs == [None]:
                            raise Exception("No valid input for tokenization!!")
                
        # Prepare few-shot samples (if needed)
        if args.use_few_shot_context_extension:
            
            # Do nothing if sample is already context-extended
            if 'already_extended' in examples and examples['already_extended'][0]=='yes':
                pass 

            # Otherwise
            else:
                # Define FS example candidates -- ensure FS examples are of same prompt type as the target example
                if 'prompt_id' in examples:
                    fs_options = fs_train.filter(lambda example: example['prompt_id']==examples['prompt_id'][0]) if examples['split']=="train" else fs_valid.filter(lambda example: example['prompt_id']==examples['prompt_id'][0])
                else: 
                    fs_options = fs_train.filter(lambda example: example['prompt_num']==examples['prompt_num'][0]) if examples['split']=="train" else fs_valid.filter(lambda example: example['prompt_num']==examples['prompt_num'][0])
                
                # Get 5-15 candidates to use as FS examples
                num_fs_ex = min(random.choice(list(range(10,30))), len(fs_options)) # if # options < 5 just take as many as possible
                random_indices = random.sample(range(len(fs_options)), num_fs_ex) 
                fs_examples = fs_options.select(random_indices)
                
                # Get FS inputs & outputs
                fs_inputs = [doc for doc in fs_examples[args.instruction_column_name]]
                fs_outputs = [ans for ans in fs_examples['answer']]

                # Format FS examples
                formatted_fs_ex = []
                if args.use_chat_template:  # format FS examples for chat models
                    for fs_input, fs_output in zip(fs_inputs, fs_outputs):
                        # 1: Format current example
                        chat_formatted_json = [
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": fs_input},
                            {"role": "assistant", "content": fs_output}
                        ]
                        chat_template_input = tokenizer.apply_chat_template(chat_formatted_json, tokenize=False, add_generation_prompt = False) 
                        # 2: Truncate if needed
                        if args.use_truncation:
                            formatted_fs_ex.append(
                                truncate_multi_doc(
                                    chat_template_input,
                                    tokenizer, 
                                    max_length=args.truncate_max_length//(num_fs_ex+1), # Truncate appropriately
                                )
                            )
                        else: 
                            formatted_fs_ex.append(chat_template_input)
                else: # Format FS examples for non chat models
                    for fs_input, fs_output in zip(fs_inputs, fs_outputs):
                        formatted_ex = f"Question:\n{fs_input}\n\nAnswer:\n{fs_output}"
                        if args.use_truncation:
                            formatted_fs_ex.append(
                                truncate_multi_doc(
                                    formatted_ex,
                                    tokenizer, 
                                    max_length=args.truncate_max_length//(num_fs_ex+1), # truncate appropriately
                                ).replace("Question: ", "Question:\n")
                            )
                        else: 
                            formatted_fs_ex.append(formatted_ex)
                    
        # Format inputs for unsloth
        if args.use_unsloth:

            if args.use_chat_template:
                chat_formatted_json = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": inputs[0]},
                    {"role": "assistant", "content": outputs[0]}
                ]
                chat_template_input = tokenizer.apply_chat_template(chat_formatted_json, tokenize=False, add_generation_prompt = False) 

                # Truncate appropriately
                if args.use_truncation:
                    if args.use_few_shot_context_extension: 
                        inputs = [
                            truncate_multi_doc(chat_template_input, tokenizer, max_length=args.truncate_max_length//(num_fs_ex+1))
                            ]
                    else: 
                        inputs = [
                            truncate_multi_doc(chat_template_input, tokenizer, max_length=args.truncate_max_length)
                            ]
                else: 
                    inputs = [chat_template_input]

                # Concatenate FS samples together with the target instruction
                if args.use_few_shot_context_extension:
                    fs_and_target_ex = formatted_fs_ex + inputs
                    inputs = ["\n".join(fs_and_target_ex)]
                return {"text": inputs}

        # Format input for causal (autoregressive) LMs
        elif "Qwen" in args.model_name or args.special_load:
            if args.use_chat_template:
                chat_formatted_json = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": inputs[0]},
                    {"role": "assistant", "content": outputs[0]}
                ]
                chat_template_input = tokenizer.apply_chat_template(chat_formatted_json, tokenize=False)
                if args.use_eot_token:
                    chat_template_input = chat_template_input.strip("\n")+"<|endoftext|>"

                # Truncate appropriately
                if args.use_truncation:
                    if args.use_few_shot_context_extension: 
                        inputs = [
                            truncate_multi_doc(chat_template_input, tokenizer, max_length=args.truncate_max_length//(num_fs_ex+1))
                            ]
                    else: 
                        inputs = [
                            truncate_multi_doc(chat_template_input, tokenizer, max_length=args.truncate_max_length)
                            ]
                else: 
                    inputs = [chat_template_input]

                # Concatenate FS samples together with the target instruction
                if args.use_few_shot_context_extension:
                    fs_and_target_ex = formatted_fs_ex + inputs
                    inputs = ["\n".join(fs_and_target_ex)]
            model_inputs = tokenizer(inputs, 
                                    max_length=args.truncate_max_length, 
                                    padding="max_length",
                                    truncation=True, 
                                )

        # Format input for seq2seq (enc-dec) LMs
        elif "t5" in args.model_name:

            # Truncate appropriately
            if args.use_truncation:
                if args.use_few_shot_context_extension: 
                    fs_templated_input = f"Question:\n{inputs[0]}\n\nAnswer:"
                    inputs = [
                        truncate_multi_doc(fs_templated_input, tokenizer, max_length=args.truncate_max_length//(num_fs_ex+1)).replace("Question: ", "Question:\n")
                        ]
                else: 
                    inputs = [
                        truncate_multi_doc(inputs[0], tokenizer, max_length=args.truncate_max_length)
                        ]
            else: 
                pass # since inputs already a list of a single element 

            # Concatenate FS samples together with the target instruction
            if args.use_few_shot_context_extension:
                fs_and_target_ex = formatted_fs_ex + inputs
                inputs = ["\n\n".join(fs_and_target_ex)]
            model_inputs = tokenizer(
                inputs, 
                max_length=args.truncate_max_length, 
                padding="max_length",  
                truncation=True, 
            )

            labels = tokenizer(
                text_target=outputs, 
                max_length=512, 
                padding="max_length",
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    ### Load model
    if args.use_unsloth:
        max_seq_length = args.truncate_max_length 
        dtype = None 
        load_in_4bit = True # Use 4bit quantization to reduce memory usage.

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        if args.fix_peft_modules:
            model = FastLanguageModel.get_peft_model(
                model,
                r = args.lora_r, 
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head",],
                lora_alpha = args.lora_alpha,
                lora_dropout = 0, 
                bias = "none",   
                random_state = 3407,
                use_rslora = False,  
                loftq_config = None, 
            )
        else:
            model = FastLanguageModel.get_peft_model(
                model,
                r = args.lora_r, 
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = args.lora_alpha,
                lora_dropout = 0, 
                bias = "none",   
                random_state = 3407,
                use_rslora = False,  
                loftq_config = None, 
            )
        if args.use_chat_template:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template = "chatml", 
                map_eos_token = True, 
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        
        config = AutoConfig.from_pretrained(args.model_name)
        config.dropout_rate = 0.05
        if args.use_peft and not args.special_load:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
            if "Qwen" in args.model_name:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name, 
                    config=config, 
                    trust_remote_code=True,
                )
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
                    lora_dropout=0,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
            elif "t5" in args.model_name:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name, 
                    load_in_8bit=True, 
                    device_map="auto",
                    config=config
                )
                lora_config = LoraConfig(
                    r=args.lora_r, 
                    lora_alpha=args.lora_alpha,
                    target_modules=["q", "v"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM
                )
                # prepare int-8 model for training
                model = prepare_model_for_kbit_training(model)

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else: 
            if "Qwen" in args.model_name:
                model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, trust_remote_code=True)
            elif "t5" in args.model_name:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)
            
    ### Prepare dataset
    train_ds_to_concatenate = []
    valid_ds_to_concatenate = []
    if args.train_num_samples and args.valid_num_samples and (args.json_dir or (args.train_json_path and args.valid_json_path)):
        if args.json_dir:
            train_path = os.path.join(args.json_dir,'train.json')
            valid_path = os.path.join(args.json_dir,'valid.json')
        else: 
            train_path = args.train_json_path
            valid_path = args.valid_json_path
        
        try:
            train_df = pd.read_json(train_path, lines=True)
            train_df['answer'] = train_df['answer'].astype(str)
            if not args.do_not_skip_na:
                train_df = train_df.dropna()
            valid_df = pd.read_json(valid_path, lines=True)
            valid_df['answer'] = valid_df['answer'].astype(str)
            if not args.do_not_skip_na:
                valid_df = valid_df.dropna()
            
            if "Unscored" not in train_path and "/data_jsons" not in train_path:
                if "RM" in train_path:
                    train_df['score'] = train_df['score'].astype(float)
                else:
                    train_df['score'] = train_df['score'].astype(int)
                if args.do_not_skip_na:
                    valid_df['score'] = valid_df['score'].fillna(5)
                if "RM" in train_path:
                    valid_df['score'] = valid_df['score'].astype(float)
                else:
                    valid_df['score'] = valid_df['score'].astype(int)
            if "/data_jsons" not in train_path:
                train_df['prompt_num'] = train_df['prompt_num'].astype(int)
            if args.do_not_skip_na:
                valid_df['prompt_num'] = valid_df['prompt_num'].fillna(2)
            if "/data_jsons" not in valid_path:
                valid_df['prompt_num'] = valid_df['prompt_num'].astype(int)

            if args.do_not_skip_na:
                # Set prompt id for anything not set yet 
                train_df['prompt_id'] = train_df.apply(lambda row: f"old_{row['prompt_num']}" if pd.isna(row['prompt_id']) else row['prompt_id'], axis=1)
                valid_df['prompt_id'] = valid_df.apply(lambda row: f"old_{row['prompt_num']}" if pd.isna(row['prompt_id']) else row['prompt_id'], axis=1)

                # Likewise put placeholder string in any empty columns as needed
                columns_to_update = ['length_spec', 'length_direction', 'format direction', 'scoring rubric ID']
                train_df[columns_to_update] = train_df[columns_to_update].fillna("")
                valid_df[columns_to_update] = valid_df[columns_to_update].fillna("")

            if args.choose_by_rank:
                train_df = train_df.sample(frac=1).nlargest(min(train_df.shape[0], args.train_num_samples), 'score')
                valid_df = valid_df.sample(frac=1).nlargest(min(valid_df.shape[0], args.valid_num_samples), 'score')

            train_ds_loaded = Dataset.from_pandas(train_df)
            valid_ds_loaded = Dataset.from_pandas(valid_df)
        
        except:
            train_df = pd.read_json(train_path, lines=True)
            train_df['answer'] = train_df['answer'].astype(str)
            valid_df = pd.read_json(valid_path, lines=True)
            valid_df['answer'] = valid_df['answer'].astype(str)
            del valid_df['__index_level_0__']

            if args.choose_by_rank:
                train_df = train_df.sample(frac=1).nlargest(min(train_df.shape[0], args.train_num_samples), 'score')
                valid_df = valid_df.sample(frac=1).nlargest(min(valid_df.shape[0], args.valid_num_samples), 'score')
                
            train_ds_loaded = Dataset.from_pandas(train_df)
            valid_ds_loaded = Dataset.from_pandas(valid_df)

        # If NOT already shuffled
        if not args.already_shuffled:
            train_ds_loaded = train_ds_loaded.shuffle(seed=args.random_seed).select(range(min(train_df.shape[0], args.train_num_samples)))
            valid_ds_loaded = valid_ds_loaded.shuffle(seed=args.random_seed).select(range(min(valid_df.shape[0], args.valid_num_samples)))
        else: 
            train_ds_loaded = train_ds_loaded.select(range(min(train_df.shape[0], args.train_num_samples)))
            valid_ds_loaded = valid_ds_loaded.select(range(min(valid_df.shape[0], args.valid_num_samples)))

        print("NUMBER OF ROWS FOR SOURCE 1", train_ds_loaded.num_rows, valid_ds_loaded.num_rows)
        print("##################################################")
        train_ds_to_concatenate.append(train_ds_loaded)
        valid_ds_to_concatenate.append(valid_ds_loaded)

    
    ### If a second dataset inputted
    if args.train_num_samples2 and args.valid_num_samples2 and (args.json_dir2 or (args.train_json_path2 and args.valid_json_path2)):
        if args.json_dir2:
            train_path2 = os.path.join(args.json_dir2,'train.json')
            valid_path2 = os.path.join(args.json_dir2,'valid.json')
        else: 
            train_path2 = args.train_json_path2
            valid_path2 = args.valid_json_path2
        
        try:
            train_df2 = pd.read_json(train_path2, lines=True)
            train_df2['answer'] = train_df2['answer'].astype(str)
            if not args.do_not_skip_na2:
                train_df2 = train_df2.dropna()
            valid_df2 = pd.read_json(valid_path2, lines=True)
            valid_df2['answer'] = valid_df2['answer'].astype(str)
            if not args.do_not_skip_na2:
                valid_df2 = valid_df2.dropna()
            
            if "Unscored" not in train_path2 and "/data_jsons" not in train_path2:
                if "RM" in train_path2:
                    train_df2['score'] = train_df2['score'].astype(float)
                else:
                    train_df2['score'] = train_df2['score'].astype(int)
                if args.do_not_skip_na2:
                    valid_df2['score'] = valid_df2['score'].fillna(5)
                if "RM" in train_path2:
                    valid_df2['score'] = valid_df2['score'].astype(float)
                else:
                    valid_df2['score'] = valid_df2['score'].astype(int)
            if "/data_jsons" not in train_path2:
                train_df2['prompt_num'] = train_df2['prompt_num'].astype(int)
            if args.do_not_skip_na2:
                valid_df2['prompt_num'] = valid_df2['prompt_num'].fillna(2)
            if "/data_jsons" not in valid_path2:
                valid_df2['prompt_num'] = valid_df2['prompt_num'].astype(int)

            if args.do_not_skip_na2:
                # Set prompt id for anything not set yet 
                train_df2['prompt_id'] = train_df2.apply(lambda row: f"old_{row['prompt_num']}" if pd.isna(row['prompt_id']) else row['prompt_id'], axis=1)
                valid_df2['prompt_id'] = valid_df2.apply(lambda row: f"old_{row['prompt_num']}" if pd.isna(row['prompt_id']) else row['prompt_id'], axis=1)

                # Likewise put placeholder string in any empty columns as needed
                columns_to_update = ['length_spec', 'length_direction', 'format direction', 'scoring rubric ID']
                train_df2[columns_to_update] = train_df2[columns_to_update].fillna("")
                valid_df2[columns_to_update] = valid_df2[columns_to_update].fillna("")

            if args.choose_by_rank2:
                train_df2 = train_df2.sample(frac=1).nlargest(min(train_df2.shape[0], args.train_num_samples2), 'score')
                valid_df2 = valid_df2.sample(frac=1).nlargest(min(valid_df2.shape[0], args.valid_num_samples2), 'score')
                
            train_ds_loaded2 = Dataset.from_pandas(train_df2)
            valid_ds_loaded2 = Dataset.from_pandas(valid_df2)

        except:
            train_df2 = pd.read_json(train_path2, lines=True)
            train_df2['answer'] = train_df2['answer'].astype(str)
            valid_df2 = pd.read_json(valid_path2, lines=True)
            valid_df2['answer'] = valid_df2['answer'].astype(str)
            del valid_df2['__index_level_0__']

            if args.choose_by_rank2:
                train_df2 = train_df2.sample(frac=1).nlargest(min(train_df2.shape[0], args.train_num_samples2), 'score')
                valid_df2 = valid_df2.sample(frac=1).nlargest(min(valid_df2.shape[0], args.valid_num_samples2), 'score')

            train_ds_loaded2 = Dataset.from_pandas(train_df2)
            valid_ds_loaded2 = Dataset.from_pandas(valid_df2)

        # If NOT already shuffled
        if not args.already_shuffled:
            train_ds_loaded2 = train_ds_loaded2.shuffle(seed=args.random_seed).select(range(min(train_df2.shape[0], args.train_num_samples2)))
            valid_ds_loaded2 = valid_ds_loaded2.shuffle(seed=args.random_seed).select(range(min(valid_df2.shape[0], args.valid_num_samples2)))
        else: 
            train_ds_loaded2 = train_ds_loaded2.select(range(min(train_df2.shape[0], args.train_num_samples2)))
            valid_ds_loaded2 = valid_ds_loaded2.select(range(min(valid_df2.shape[0], args.valid_num_samples2)))

        print("NUMBER OF ROWS FOR SOURCE 2", train_ds_loaded2.num_rows, valid_ds_loaded2.num_rows)
        print("##################################################")
        train_ds_to_concatenate.append(train_ds_loaded2)
        valid_ds_to_concatenate.append(valid_ds_loaded2)

    if not args.already_shuffled:
        train_ds = concatenate_datasets(train_ds_to_concatenate).shuffle(seed=args.random_seed)
        valid_ds = concatenate_datasets(valid_ds_to_concatenate).shuffle(seed=args.random_seed)
    else: 
        train_ds = concatenate_datasets(train_ds_to_concatenate)
        valid_ds = concatenate_datasets(valid_ds_to_concatenate)
    
    print("NUM_TRAIN_SAMPLES:",train_ds.num_rows)
    print("NUM_VALID_SAMPLES:",valid_ds.num_rows)

    if args.save_ds_only:
        train_ds.to_json(args.train_save_path) 
        valid_ds.to_json(args.valid_save_path) 

    # If using few-shot samples
    if args.use_few_shot_context_extension:
        # Define FS example pool -- first 90% of the data
        fs_train = train_ds.select(range(int(len(train_ds)*(1-args.fs_proportion))))
        fs_valid = valid_ds.select(range(int(len(valid_ds)*(1-args.fs_proportion))))
        
        # Add new column to the current ds to indicate what split each example is in
        train_col = ["train"] * len(train_ds)
        valid_col = ["valid"] * len(valid_ds)
        train_ds = train_ds.add_column("split", train_col)
        valid_ds = valid_ds.add_column("split", valid_col)

        # Isolate last 10% of train/val samples for augmentation with few-shot samples
        train_ds = train_ds.select(range(len(train_ds)-int(len(train_ds)*args.fs_proportion), len(train_ds)))
        valid_ds = valid_ds.select(range(len(valid_ds)-int(len(valid_ds)*args.fs_proportion), len(valid_ds)))

        print("##################################################")
        print("NUMBER OF ROWS FOR FEW-SHOT SELECTION", train_ds.num_rows, valid_ds.num_rows)

    # Format dataset as dataset dict for HF
    dataset = DatasetDict()
    dataset["train"] = train_ds
    dataset["validation"] = valid_ds

    ### Process dataset 
    tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=args.tokenization_batch_size)
    if "Qwen" in args.model_name or args.special_load:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # set mlm = False so it does Causal LM and not Masked LM
    elif "t5" in args.model_name:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    ### Training setup
    compute_fxn = None

    # Select correct training arguments & trainer type
    if "Qwen" in args.model_name or args.special_load:
        trainer_fxn = Trainer
    elif "t5" in args.model_name:
        trainer_fxn = Seq2SeqTrainer

    if args.use_unsloth:
        training_args = TrainingArguments(
                output_dir=checkpoint_dir,
                evaluation_strategy="steps",
                save_strategy="steps",
                per_device_train_batch_size=args.per_train_batch_size, # 2
                per_device_eval_batch_size=args.per_eval_batch_size, # 4
                gradient_accumulation_steps=args.grad_acc_steps,
                max_steps=args.num_train_steps,
                save_steps=args.save_steps, 
                eval_steps=args.eval_steps, 
                warmup_ratio=args.warmup_ratio,
                logging_steps=5,
                optim="adamw_torch",
                learning_rate=args.lr, 
                lr_scheduler_type=args.lr_scheduler_type, 
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                weight_decay = args.weight_decay,
                save_total_limit=args.save_total_limit,
                load_best_model_at_end=False,
                resume_from_checkpoint=args.checkpoint,
                gradient_checkpointing=args.grad_checkpointing,
                push_to_hub=False,
                report_to="wandb",
            )
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False, 
            args = training_args,
        )
    
    else: # if NOT using unsloth
        if "Qwen" in args.model_name or args.special_load:
                training_args = TrainingArguments(
                output_dir=checkpoint_dir,
                evaluation_strategy="steps",
                save_strategy="steps",
                weight_decay=args.weight_decay,
                adam_beta1=0.9,
                adam_beta2=0.98,
                adam_epsilon=1e-06,
                learning_rate=args.lr,
                lr_scheduler_type=args.lr_scheduler_type,
                optim="adamw_torch",
                per_device_train_batch_size=args.per_train_batch_size,
                per_device_eval_batch_size=args.per_eval_batch_size, 
                gradient_accumulation_steps=args.grad_acc_steps,
                max_steps=args.num_train_steps,
                save_steps=args.save_steps, 
                eval_steps=args.eval_steps, 
                warmup_ratio=args.warmup_ratio,
                logging_steps=5,
                save_total_limit=args.save_total_limit,
                load_best_model_at_end=True,
                fp16=args.fp16,
                bf16=args.bf16,
                push_to_hub=False,
                report_to="wandb",
                gradient_checkpointing=args.grad_checkpointing,
            )
        elif "t5" in args.model_name:
            training_args = Seq2SeqTrainingArguments(
                output_dir=checkpoint_dir,
                evaluation_strategy="steps",
                save_strategy="steps",
                weight_decay=args.weight_decay,
                adam_beta1=0.9,
                adam_beta2=0.98,
                adam_epsilon=1e-06,
                learning_rate=args.lr,
                lr_scheduler_type=args.lr_scheduler_type,
                optim="adamw_torch",
                per_device_train_batch_size=args.per_train_batch_size,
                per_device_eval_batch_size=args.per_eval_batch_size, 
                gradient_accumulation_steps=args.grad_acc_steps,
                max_steps=args.num_train_steps,
                save_steps=args.save_steps, 
                eval_steps=args.eval_steps, 
                warmup_ratio=args.warmup_ratio,
                logging_steps=5,
                save_total_limit=args.save_total_limit,
                load_best_model_at_end=True,
                predict_with_generate=True,
                generation_max_length=512,
                generation_num_beams=1,
                fp16=args.fp16,
                push_to_hub=False,
                report_to="wandb",
                gradient_checkpointing=args.grad_checkpointing,
            )
        trainer = trainer_fxn(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_fxn,
        )

    if args.use_peft or args.no_gpu or (not args.use_peft and int(os.environ["LOCAL_RANK"])==0): 
        wandb.watch(model, log='gradients', log_freq=1)
    if args.use_unsloth:
        trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("FINAL MODEL SAVED TO:",model_save_dir)

    if args.use_multinews or args.use_hqa:
        test_results = trainer.evaluate(
            tokenized_dataset["test"], 
            metric_key_prefix="test",
            max_length=512,
            num_beams=1, # use greedy decoding instead of beam search
        )
        print(test_results)
    
    if args.use_peft or args.no_gpu or (not args.use_peft and int(os.environ["LOCAL_RANK"])==0):
        if args.use_multinews or args.use_hqa or args.use_qmdscnn or args.use_mxss or args.use_wikihop:
            wandb.log(test_results)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of path to appropriate train/val/test.json folder
    parser.add_argument("--wandb_project_name", 
                        type=str, 
                        default=None,
                        ) 
    parser.add_argument("--model_name", 
                        type=str,
                        default="google/flan-t5-base"
                        ) 
    parser.add_argument("--short_model_name", 
                        type=str,
                        default=None,
                        ) 
    parser.add_argument("--wandb_group", 
                        type=str,
                        ) 
    
    # directory to save instruction-tuned checkpoints
    parser.add_argument("--output_dir",  
                        type=str,
                        default="./IT"
                        ) 

    # training args
    parser.add_argument("--lr", 
                        type=float, 
                        default=5e-4,
                        ) 
    parser.add_argument("--lr_scheduler_type", 
                        type=str, 
                        default="constant",
                        ) 
    parser.add_argument("--optimizer", 
                        type=str, 
                        default="adafactor",
                        ) 
    parser.add_argument("--warmup_ratio", 
                        type=float, 
                        default=0,
                        ) 
    parser.add_argument("--weight_decay", 
                        type=float, 
                        default=0.001,
                        ) 

    parser.add_argument("--save_total_limit", type=int, default=7) 
    parser.add_argument("--per_train_batch_size", type=int, default=2) 
    parser.add_argument("--per_eval_batch_size", type=int, default=2) 
    parser.add_argument("--grad_acc_steps", type=int, default=4) 
    parser.add_argument("--save_steps", type=int) 
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int)

    parser.add_argument("--use_unsloth", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--fp16", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--bf16", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--use_peft", 
                        action="store_true",
                        default=False,
                        ) 
    parser.add_argument("--no_gpu", 
                        action="store_true",
                        default=False,
                        )   
    parser.add_argument("--resume_from_checkpoint", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--grad_checkpointing", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--checkpoint", 
                        default=None, 
                        ) 
    parser.add_argument("--wandb_run_id", 
                        default=None, 
                        )
    parser.add_argument("--lora_r", 
                        type=int,
                        default=16,
                        )  
    parser.add_argument("--lora_alpha", 
                        type=int,
                        default=32,
                        )  

    # if only running to generate and save dataset
    parser.add_argument("--save_ds_only", 
                        action="store_true",
                        default=False,
                        )                     

    # dataset preparation args (for OUR instructions)
    parser.add_argument("--instruction_column_name",
                        type=str, 
                        default="instruction",
                        help="which key in the input jsons to use as the instruction inputs -- e.g. instruction_with_length_direction vs formatted_instruction_with_length_direction",
                        ) 
    parser.add_argument("--train_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--valid_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--test_num_samples", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--train_num_samples2", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--valid_num_samples2", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--test_num_samples2", 
                        type=int, 
                        default=None,
                        ) 
    parser.add_argument("--truncate_max_length", 
                        type=int, 
                        default=4096,
                        ) 
    parser.add_argument("--use_eot_token", 
                        action="store_true",
                        default=False,
                        )  
    parser.add_argument("--use_chat_template", 
                        action="store_true",
                        default=False,
                        )  

    # tokenization arguments
    parser.add_argument("--tokenization_batch_size", 
                        type=int, 
                        default=1000,
                        ) 
    parser.add_argument("--use_truncation", 
                        action="store_true",
                        default=False,
                        ) 

    # dataset selection
    # PROVIDE EITHER:
    parser.add_argument("--json_dir", type=str)
    # OR (both below):
    parser.add_argument("--train_json_path", type=str, default=None)
    parser.add_argument("--valid_json_path", type=str, default=None)

    # PROVIDE EITHER:
    parser.add_argument("--json_dir2", type=str)
    # OR (both below):
    parser.add_argument("--train_json_path2", type=str, default=None)
    parser.add_argument("--valid_json_path2", type=str, default=None)

    
    parser.add_argument("--train_save_path", type=str, default=None)
    parser.add_argument("--valid_save_path", type=str, default=None)
    

    parser.add_argument("--random_seed", 
                        type=int, 
                        default=42,
                        ) 
    parser.add_argument("--use_few_shot_context_extension", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--already_shuffled", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--choose_by_rank", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--choose_by_rank2", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--do_not_skip_na", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--do_not_skip_na2", 
                        action="store_true",
                        default=False, 
                        ) 
    parser.add_argument("--fs_proportion", 
                        type=float, 
                        default=0.1,
                        )  

    parser.add_argument("--fix_peft_modules", 
                        action="store_true",
                        default=False,
                        )  
    parser.add_argument("--special_load", 
                        action="store_true",
                        default=False,
                        )  
                    
    args = parser.parse_args()

    main(args)

