import argparse
import os
import json
import pandas as pd

def main(args):

    get_length_direction = {
        "2": "Answer briefly in 1-2 sentences.",
        "3": "Answer 'yes' or 'no'",
        "4": "Answer 'yes' or 'no'",
        "5": "Answer with a single word or brief phrase.",
        "6": "Answer with at least 5 sentences.",
        "7": "Answer with at most 5 sentences.",
    }
    
    for split in [
        'train.json', 
        'valid.json',
        ]:
        print("On", split)

        data_df = pd.read_json(os.path.join(args.data_json_dir, split), lines=True)

        for index, row in data_df.iterrows():
            instruction = row['instruction']

            try:
                prompt_num = int(row['prompt_num'])
            except: 
                prompt_num = args.prompt_num

            length_direction = get_length_direction[str(int(prompt_num))]

            revised_instrxn = f"{instruction} {length_direction}"

            data_df.at[index, 'instruction'] = revised_instrxn

        # save to new filename
        filename = split[:-5]+"_with_length_direction.json"
        output_file = os.path.join(args.data_json_dir, filename)
        data_df.to_json(output_file, orient='records',lines=True)
        
        print(f"Finished {split}!")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_json_dir", 
                        type=str,
                        help="Name of path to appropriate train/val/test.json folder"
                        )
    parser.add_argument("--prompt_num", 
                        type=str,
                        default=-1,
                        help="Optional specification of prompt number. Defaults to -1 if not already recorded in the data jsons."
                        )
    args = parser.parse_args()

    main(args)


