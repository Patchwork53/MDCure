import re
import json
from openai import OpenAI
import rm_prompt as prompt

def gpt_preference():
    data = []
    file_bases = [
        "instruction_clarity",
        "instruction_comprehensive",
        "instruction_conciseness",
        "instruction_multi_doc",
        "instruction_relevance"
    ]

    for file in file_bases:
        path = f"./data/{file}.jsonl"
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))

    client = OpenAI()
    output_path = "./data/ratings_output.jsonl"
    for i, d in enumerate(data):
        context = d.get("context", "")
        instruction = d.get("generation", "")
        
        formatted_input = prompt.gpt_preference_gen.format(context=context, instruction=instruction)
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional annotator."},
                    {"role": "user", "content": formatted_input}
                ]
            )

            gpt_output = completion.choices[0].message.content

            rating = {
                "context": context,
                "instruction": instruction,
                "response": gpt_output
            }

            with open(output_path, 'a') as f:
                f.write(json.dumps(rating) + '\n')

            print(f"Generated ratings for instruction {i}")

        except Exception as e:
            print(f"Error processing data entry {i}: {e}")

def parse_scores(response_text):
    pattern = r"Relevance:\s*\[?(\d+)\]?\s*.*Coherence\s*&\s*Factuality:\s*\[?(\d+)\]?\s*.*Creativity:\s*\[?(\d+)\]?\s*.*Context\s*Integration:\s*\[?(\d+)\]?\s*.*Inter-Document\s*Relationships:\s*\[?(\d+|N/A)\]?\s*.*Complexity:\s*\[?(\d+)\]?"
    
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return {
            "Relevance": int(match.group(1)),
            "Coherence & Factuality": int(match.group(2)),
            "Creativity": int(match.group(3)),
            "Context Integration": int(match.group(4)),
            "Inter-Document Relationships": int(match.group(5)),
            "Complexity": int(match.group(6))
        }
    else:
        raise ValueError(f"Could not parse response: {response_text}")

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            
            try:
                parsed_scores = parse_scores(data.get('response', ''))
                data.update(parsed_scores)
                del data['response']
                
                outfile.write(json.dumps(data) + '\n')
            except ValueError as e:
                print(f"Error processing line: {e}")

if __name__ == "__main__":
    gpt_preference()
    process_jsonl_file("./data/ratings_output.jsonl", "./data/parsed_ratings.jsonl")
