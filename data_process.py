import json
import os
import argparse
from detoxify import Detoxify

""" 
adhoc processing of challenging_prompts to reformat it to look like nontoxic_prompts-10k

adhoc processing of challenging generations to reformat it to look like challenging prompts again

adhoc scoring of the generations with detoxify
"""

def reformat_challenging_prompts():
    """
    Reformats challenging_prompts.jsonl to match the format of nontoxic_prompts-10k.jsonl
    Only keeps the 'prompt' field and wraps it in a 'text' field.
    """
    input_file = "data/prompts/challenging_prompts.jsonl"
    output_file = "data/prompts/challenging_prompts_reformatted.jsonl"
    
    reformatted_data = []
    
    # Read and reformat each line
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Create new format with just prompt.text
            reformatted = {
                "prompt": {
                    "text": data["prompt"]
                }
            }
            reformatted_data.append(reformatted)
    
    # Write reformatted data to new file
    with open(output_file, 'w') as f:
        for item in reformatted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Reformatted {len(reformatted_data)} prompts")
    print(f"Saved to {output_file}")

def reformat_challenging_generations(input_file, output_file):
    """ 
    Reformat challenging generations back to just:
    {
        "prompt": "...",
        "generation": "..."
    }
    just take the first one
    """

    reformatted_data = []
    
    # Read and reformat each line
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Create new format with just prompt.text
            reformatted = {
                "prompt": data['prompt']['text'],
                "generation": data['generations'][0]['text']
            }
            reformatted_data.append(reformatted)
    
    # Write reformatted data to new file
    with open(output_file, 'w') as f:
        for item in reformatted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Reformatted {len(reformatted_data)} generations")
    print(f"Saved to {output_file}")

def get_toxicity_scores(input_file, output_file):
    """ 
    input: jsonl of form {'prompt': '...', 'generation': '...'}
    output: jsonl of form {'prompt': '...', 'generation': '...', 'toxicity': {...}}
    """
    toxicity_model = Detoxify('unbiased')

    output_data = []

    avg_tox = 0.0
    
    # Read and reformat each line
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            output_data_row = {
                "prompt": data['prompt'],
                "generation": data['generation'],
                "toxicity": {k: float(v) for k, v in toxicity_model.predict(data['generation']).items()} # float64
            }
            avg_tox += output_data_row['toxicity']['toxicity']
            output_data.append(output_data_row)
    
    # Write reformatted data to new file
    with open(output_file, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Scored {len(output_data)} generations")
    print(f"Saved to {output_file}")
    return avg_tox/len(output_data)


if __name__ == "__main__":
    model = "gemma-2-2b" # gpt2-large, etc. it's what you name it NOT the hf name
    # reformat_challenging_prompts()

    """
    reformat generations
    """
    # input_file = f"logs/detoxification-{model}/predictions.jsonl"
    # output_file = f"logs/detoxification-{model}/predictions_reformatted.jsonl"
    # reformat_challenging_generations(input_file, output_file)
    """ 
    score generations
    """
    # input_file = f"logs/detoxification-{model}/predictions_reformatted.jsonl"
    # output_file = f"logs/detoxification-{model}/predictions_scored.jsonl"
    # avg_tox = get_toxicity_scores(input_file, output_file)
    # print(f'avg_tox: {avg_tox}')
