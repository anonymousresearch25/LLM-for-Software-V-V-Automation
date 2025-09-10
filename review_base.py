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

# for time & date
from datetime import datetime

# imports for custom rag pipeline design
from main import (
    setup_rag_pipeline, 
    csv_to_json,
    save_to_jsonl,
    select_output_code_stack,
    load_code_stack_dataset,
    generate_base_prompt,
    knowledge_base_loader, 
    query,
    progress_tracker
)

# argument parser (from testeval)
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='leetcode')
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1' ,'meta-llama/Llama-3.1-8B-Instruct', 'tinyllama', 'deepseek-r1:1.5b'])
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=256)
    # new arg for rag config
    parser.add_argument("--rag", type=bool, default=False, help='use RAG or not')
    # new arg for hf rag pipeline
    parser.add_argument("--hf_provider", type=str, default="fireworks-ai")
    return parser.parse_args()

# testing
if __name__ == "__main__":
    # parse args
    args=parse_args()

    # load model access
    os.environ["HUGGINGFACE_TOKEN"] = open("text_files/token.txt").read().strip()
    os.environ["OPENAI_API_KEY"] = open("text_files/key.txt").read().strip()

    # save output
    output_dir = Path('predictions')
    
    # load dataset
    converted_dataset = csv_to_json("data/code_stack/datasets/bug_in_the_code_stack_alpaca_dataset.csv", "data/csv_to_json/converted.json")
    dataset = load_code_stack_dataset("data/csv_to_json/")

    if args.rag:
        print("Using RAG for code review...")
        # setup demo pipeline
        if args.model.startswith("gpt"):
            rag_pipeline = setup_rag_pipeline(provider="openai")
        elif "/" in args.model:
            rag_pipeline = setup_rag_pipeline(provider="hf", hf_provider=args.hf_provider)
        else:
            rag_pipeline = setup_rag_pipeline(provider="ollama")

        # load knowledge base
        docs = knowledge_base_loader.load_json_from_github(
            github_url="https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json",
            content_field="prompt",
            id_field="task_id",
            metadata_fields=["code", "test_list"],
        )

        # add knowledge base to the pipeline
        rag_pipeline.add_knowledge_base(docs)

        # using tqdm progress bar for dataset
        print('Model:', args.model)
        with tqdm(total=len(dataset), ncols=80) as pbar:
            # iterate through each data in the dataset
            for data in dataset:
                # randomly select one output from task
                type, code = select_output_code_stack(data)

                # create query object
                demo_query = query(
                    text=str(code),
                    context_type="bug_one_line",
                )

                # run the RAG pipeline to get a single response
                single_test_result = rag_pipeline.first_round_response(demo_query)

                # store the llm response and the bug type as a dictionary
                single_test_result = {
                    "bug_type": type,
                    "response": str(single_test_result.answer),
                }

                # breakup console
                print("\n<<<<----------------------------------------->>>>\n")

                # save the results to a JSONL file
                save_to_jsonl(single_test_result, output_dir / f'{datetime.today().strftime("%m-%d")}_code_review_rag_{args.model}.jsonl')

                # update progress bar
                pbar.update(1)

        print(f"\nProcessing completed in {time.time() - pbar.start_t:.2f} seconds")

    else:
        # using tqdm progress bar for dataset
        print('Model:', args.model)
        with tqdm(total=len(dataset), ncols=80) as pbar:
            # iterate through each data in the dataset
            for data in dataset:
                # randomly select one output from task
                type, code = select_output_code_stack(data)

                # create query object
                demo_query = query(
                    text=str(code),
                    context_type="bug_one_line",
                )

                # get model type depending on whether args.model starts with gpt or not
                if args.model.startswith("gpt"):
                    from main import open_AI
                    llm = open_AI(model=args.model)
                elif "/" in args.model:
                    from main import hf
                    llm = hf(model_name=args.model)
                else:
                    from main import ollama_class
                    llm = ollama_class(model=args.model)

                prompt = generate_base_prompt(demo_query)
                
                single_test_result = llm.generate_response(prompt)

                # store the llm response and the bug type as a dictionary
                single_test_result = {
                    "bug_type": type,
                    "response": str(single_test_result),
                }

                # breakup console
                print("\n<<<<----------------------------------------->>>>\n")

                # save the results to a JSONL file
                save_to_jsonl(single_test_result, output_dir / f'{datetime.today().strftime("%m-%d")}_code_review_{args.model}.jsonl')

                # update progress bar
                pbar.update(1)

        print(f"\nProcessing completed in {time.time() - pbar.start_t:.2f} seconds")