# import necessary libraries

# for path
from pathlib import Path

# for progress bar
import time
from tqdm import tqdm

# for environment variables
import os

# for argument parsing
from argparse import ArgumentParser

# for handling JSON data
import json

# for prompt formatting
import textwrap

# imports for main functions
from main import (
    read_jsonl, 
    save_to_jsonl
)

# argument parser (from testeval)
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default='code_review.jsonl')
    parser.add_argument("--output_dir", type=str, default=Path('results'))
    parser.add_argument("--output_file", type=str, default='eval_base.txt')
    parser.add_argument("--model", type=str, default='gpt-4.1', choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gtp-4.1'])
    parser.add_argument("--sorted", type=bool, default = False)
    return parser.parse_args()

# function to evaluate review data
def eval_review(data: str, model_name: str):
    bug_type = data.get('bug_type','')
    response = data.get('response', '')

    prompt = f"""
    Please determine whether the provided response accurately matches the bug type. We know there is a match if the response correctly acknowledges the issue.
    State only a one-word lowercase answer of 'match' for a correct labeling and 'mismatch' for an incorrect labeling.

    Possible output types:
    ----
    'output': bug-free
    'output_missing_colon': missing colon
    'output_missing_parenthesis': missing parenthesis
    'output_missing_quotation': missing quotation
    'output_missing_comma': missing comma
    'output_mismatched_quotation': mismatched quotation
    'output_mismatched_bracket': mismatched bracket
    'output_keywords_as_identifier': usage of a python keyword as a standard variable name
    ----

    We can only say 'this code is visibly bug-free' if the output type is 'output'. No exceptions.
    
    Labeled bug type:
    ----
    {bug_type}
    ----

    Given response:
    ----
    {response}
    ----
    """

    # get model type depending on whether args.model starts with gpt or not
    if model_name.startswith("gpt"):
        from main import open_AI
        llm = open_AI(model=model_name)
    else:
        from main import llama
        llm = llama()

    llm_interaction = llm.generate_response(textwrap.dedent(prompt))

    # store the llm response and the output info as a dictionary
    output_eval = {
        "llm truth label": str(llm_interaction),
        "labeled bug type": bug_type,
        "initial llm response": response,
    }

    return output_eval

# function to count results
def match_count(file_path: str):    
    # open file and read all lines
    with open(file_path) as file:
        lines = file.readlines()

    # initialize counts
    match_count = 0
    mismatch_count = 0

    # track the types of mismatches
    mismatch_type = {
        "output":0, 
        "output_missing_colon":0, 
        "output_missing_parenthesis":0, 
        "output_missing_quotation":0,
        "output_missing_comma":0,
        "output_mismatched_quotation":0,
        "output_mismatched_bracket":0,
        "output_keywords_as_identifier":0
    }

    with tqdm(total=len(lines), ncols=80) as pbar:
        # iterate through each line in the file
        for line in lines:
            # convert data to dict form
            data = json.loads(line.strip())

            # get llm truth label
            label = data.get('llm truth label','')

            # get labeled bug type
            bug_type = data.get('labeled bug type', '')

            # increment match count
            if label.lower() == "match":
                match_count += 1
            
            # increment mismatch count
            elif label.lower() == "mismatch":
                mismatch_count += 1
                # track bug type
                if bug_type in mismatch_type:
                    mismatch_type[bug_type] += 1

            # throw error       
            else:
                print("Error: unsupported label")

            # update progress bar
            pbar.update(1)

    # keep track of each count type
    return match_count, mismatch_count, mismatch_type

# function to format bug type counts
def format_mismatch_type_count(mismatch_type: dict, mismatches: int, sort: bool):
    # init list
    tracker = []

    # iterate through dict
    for key, value in mismatch_type.items():
        # calculate percentage of total mismatches
        rate = ((value / mismatches)) * 100

        # round result
        rate = str(round(rate, 2))

        # if not 1 
        if value == 0 or value > 1:
            string = (f"{rate}%: {value} mismatches for bug type: '{key}'")
            tracker.append(string)
        # if 1
        elif value == 1:
            string = (f"{rate}%: {value} mismatch for bug type: '{key}'")
            tracker.append(string)
        # invalid case
        else:
            string = (f"Invalid count of '{key}' mismatches")
            tracker.append(string)

    # option to return sorted list
    if sort == True:
        # sort list by percentage (descending)
        sorted_tracker = sorted(tracker, key=lambda x: float(x.split('%')[0]), reverse=True)

        # return sorted list
        return sorted_tracker

    # alternatively, return unsorted list
    else:
        return tracker

if __name__ == "__main__":
    # parse args
    args=parse_args()

    # load model access
    os.environ["HUGGINGFACE_TOKEN"] = open("text_files/token.txt").read().strip()
    os.environ["OPENAI_API_KEY"] = open("text_files/key.txt").read().strip()

    print(f"Evaluating LLM code review performance...")

    # get data
    data = read_jsonl(Path('predictions') / args.path)

    print('Model:', args.model)
    with tqdm(total=len(data), ncols=80) as pbar:

        # evaluate data
        for datum in data:
            # run evaluation
            eval = eval_review(datum, args.model)

            # save results
            save_to_jsonl(eval, args.output_dir / args.output_file)

            # update progress bar
            pbar.update(1)
    
    # printing to console
    print("\nFinding accuracy...")

    # get count of matches & mismatches
    matches, mismatches, mismatch_type = match_count(args.output_dir / args.output_file)

    # compute accuracy
    accuracy = (matches / (matches + mismatches)) * 100

    # round result
    accuracy = str(round(accuracy, 2))

    # compute mismatch type count
    mismatch_types = format_mismatch_type_count(mismatch_type, mismatches, args.sorted)

    # iterate through mismatch types print mismatch types
    for item in mismatch_types:
        print(item)

    # printing to console
    print(f"\n{accuracy}% accuracy with {matches} matches and {mismatches} mismatches")
    print(f"\nProcessing completed in {time.time() - pbar.start_t:.2f} seconds\n")