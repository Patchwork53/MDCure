from pydantic import BaseModel

import re
import json
import random
import argparse
from typing import Any, Dict, List

from distilabel.pipeline import Pipeline

from distilabel.llms import vLLM, OpenAILLM, MistralLLM

from distilabel.steps import Step, StepInput, make_generator_step, CombineColumns, KeepColumns
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import PrometheusEval, TextGeneration
from distilabel.steps.tasks.typing import ChatType

from rich import traceback
traceback.install(show_locals=False)

import rm_prompt as prompt

class User(BaseModel):
    Instruction: str
    Answer: str

class InstructionGeneration(TextGeneration):
    @property
    def inputs(self) -> List[str]:
        return ["context", "cluster_id"]

    @property
    def outputs(self) -> List[str]:
        return ["model_name", "instruction", "generation", "cluster_id"]

    def format_input(self, input: List[Dict[str, Any]]) -> "ChatType":
        return [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.instruct_prompt.format(**input)}
        ]

    def format_output(self, output: str | None, input: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(output, str):
            try:
                output_dict = json.loads(output)
            except json.JSONDecodeError:
                output_dict = {"Instruction": output}
        else:
            output_dict = output
        
        cluster_id = input.get("cluster_id", None)
        if cluster_id is None:
            raise ValueError("Missing 'cluster_id' in input")

        return {
            "model_name": self.llm, 
            "instruction": prompt.instruct_prompt.format(**input), 
            "generation": output_dict.get("Instruction", ""),  # Safely access the "Instruction" key
            "clutser_id": cluster_id
        }

class LLMParsing(Step):
    @property
    def inputs(self) -> List[str]:
        return ["generation"]

    @property
    def outputs(self) -> List[str]:
        return ["generation"]

    def process(self, *inputs: StepInput) -> StepOutput:
        data_list = inputs[0]

        for input_dict in data_list:
            context = input_dict.get("context")
            generation = input_dict.get("generation")

            # Look for the "instruction:" or "Instruction:" pattern in the string
            match = re.search(r'(?:^|\n)(instruction|Instruction)\s*:\s*["\']?(.+?)["\']?(?:,|\n|$)', generation, re.DOTALL)
            if match:
                instruction = match.group(2).strip()
                if args.rubric in ['relevance', 'comprehensive', 'multi_doc']:
                    input_dict["generation"] = f"Context: {context}\n\nInstruction: {instruction}"
                else:
                    input_dict["generation"] = f"Instruction: {instruction}"
            else:
                if args.rubric in ['relevance', 'comprehensive', 'multi_doc']:
                    input_dict["generation"] = f"Context: {context}\n\n{generation}"
                else:
                    input_dict["generation"] = generation

        yield data_list  

def parse_jsonl_to_dict(file_path, max_context_length=24576):
    result = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            instruction_parts = data['instruction'].split('\n\n')
            if len(instruction_parts) >= 2:
                context, instruction = "\n\n".join(instruction_parts[:-1]), instruction_parts[-1]
            else:
                context = ""
                instruction = instruction_parts[0]
                print(f"Failed to parse context for {data['cluster_id']}")
            
            if len(context) > max_context_length:
                context = context[:max_context_length]

            entry = {
                'context': context,
                'instruction': instruction,
                'answer': data['answer'],
                'cluster_id': data['cluster_id'],
            }
            result.append(entry)
    
    return result

def build_ppo_reward_dataset(eval_rubric):
    with Pipeline(name=f"reward-model-{args.rubric}", cache_dir="./data") as pipeline:
        loader = make_generator_step(sampled_data, output_mappings={
            "context": "context",
            "cluster_id": "cluster_id"
        })

        gpt_task = InstructionGeneration(
            name="instruction-gen-gpt",
            llm=OpenAILLM(
                model="gpt-4o-mini",
                api_key=OPENAI_API,
            ),
        )

        mistral_task = InstructionGeneration(
            name="instruction-gen-mistral-7B",
            llm=MistralLLM(
                model="open-mistral-7b",
                api_key=MISTRAL_API,
            ),
        )

        gpt_parse = LLMParsing(
            name="gpt-parse"
        )

        mistral_parse = LLMParsing(
            name="mistral-parse"
        )

        combine_columns = CombineColumns(
            name="combine_columns",
            columns=["generation", "model_name"],
            output_columns=["generations", "generation_models"],
        )

        prometheus = PrometheusEval(
            name="prometheus",
            llm=vLLM(
                model="prometheus-eval/prometheus-7b-v2.0",
                chat_template="[INST] {{ messages[0]['content'] }}\\n{{ messages[1]['content'] }}[/INST]",
                extra_kwargs={
                    "max_model_len": 24576, 
                },
            ),
            mode="relative",
            rubric="custom",
            rubrics={"custom": eval_rubric},
            reference=False,
            num_generations=1,
            group_generations=False,
        )
        
        keep_columns = KeepColumns(
            name="keep_columns",
            columns=["instruction", "generations", "feedback", "result", "model_name", "cluster_id"],
        )

        # Connect the components in the pipeline
        loader.connect(gpt_task)
        loader.connect(mistral_task)

        gpt_task.connect(gpt_parse)
        mistral_task.connect(mistral_parse)

        gpt_parse.connect(combine_columns)
        mistral_parse.connect(combine_columns)
        
        combine_columns.connect(prometheus)
        prometheus.connect(keep_columns)

        distiset = pipeline.run(
            parameters={
                "load_data_from_dicts_0": {
                },
                "instruction-gen-gpt": {
                    "llm": {
                        "generation_kwargs": {
                            "temperature": 0.7,
                            "max_new_tokens": 1024,
                        },
                    }
                },
                "instruction-gen-mistral-7B": {
                    "llm": {
                        "generation_kwargs": {
                            "temperature": 0.7,
                            "max_new_tokens": 1024,
                        }
                    }
                },
                "prometheus": {
                    "llm": {
                        "generation_kwargs": {
                            "temperature": 0.7,
                            "max_new_tokens": 1024,
                        },
                    },
                },
            },
        )

        df = distiset['default']['train'].to_pandas()
        df.to_json(f'./data/instrucion_{args.rubric}.jsonl', orient='records', lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Prometheus PPO Reward Model pipeline')
    parser.add_argument('--rubric', 
                        choices=['relevance', 'clarity', 'comprehensive', 'conciseness', 'multi_doc'], 
                        help='Choose the evaluation rubric to use')
    parser.add_argument('--num_sample', 
                        type=int, 
                        help='Number of samples to process')
    parser.add_argument('--file_path',
                        type=str, 
                        help='File path for the documents')
    parser.add_argument('--openai_key',
                        type=str)
    parser.add_argument('--mistral_key',
                        type=str)
    args = parser.parse_args()

    OPENAI_API = args.openai_key
    MISTRAL_API = args.mistral_key

    rubric_mapping = {
        "relevance": prompt.relevance_appropriateness_rubric,
        "clarity": prompt.clarity_specificity_rubric,
        "comprehensive": prompt.comprehensiveness_rubric,
        "conciseness": prompt.conciseness_rubric,
        "multi_doc": prompt.multi_document_rubric
    }

    selected_rubric = rubric_mapping[args.rubric]
    num_sample = args.num_sample

    max_context_length = 16384
    jsonl_file_path = args.file_path 
    raw_data = parse_jsonl_to_dict(jsonl_file_path, max_context_length)
    sampled_data = random.sample(raw_data, num_sample)

    build_ppo_reward_dataset(selected_rubric)