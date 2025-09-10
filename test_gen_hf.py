# import necessary libraries

# for path
from pathlib import Path

# for progress bar
from tqdm import tqdm

# for environment variables
import os

# for argument parsing
from argparse import ArgumentParser

# for llama
import transformers
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

# for time & date
from datetime import datetime

# imports for custom rag pipeline design
from main import (
    setup_rag_pipeline, 
    knowledge_base_loader, 
    query,
    test_eval_task,
    progress_tracker
)

# testeval (benchmark model) config
access_token=os.getenv("HUGGINGFACE_TOKEN")

# imports for testeval
from data_utils import(
    read_jsonl,
    write_jsonl,
    add_lineno
)

# original testeval arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='leetcode')
    parser.add_argument("--model", type=str, default='codellama/CodeLlama-7b-Instruct-hf')
    parser.add_argument("--num_tests", type=int, default=10, help='number of tests generated per program')
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument("--max_tokens", type=int, default=256)
    # new token for rag config
    parser.add_argument("--rag", type=bool, default=False, help='use RAG or not')
    return parser.parse_args()


model_list=['codellama/CodeLlama-7b-Instruct-hf','codellama/CodeLlama-13b-Instruct-hf','codellama/CodeLlama-34b-Instruct-hf',
            'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'bigcode/starcoder2-15b-instruct-v0.1',
            'google/gemma-1.1-2b-it', 'google/gemma-1.1-7b-it'
            'google/codegemma-7b-it',
            'deepseek-ai/deepseek-coder-1.3b-instruct', 'deepseek-ai/deepseek-coder-6.7b-instruct',
            'deepseek-ai/deepseek-coder-33b-instruct',
            'mistralai/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.3'
            'Qwen/CodeQwen1.5-7B-Chat'
            ]

#models do not support system message
models_nosys=['google/gemma-1.1-7b-it',
            'bigcode/starcoder2-15b-instruct-v0.1',
            'mistralai/Mistral-7B-Instruct-v0.3']


# original testeval functions
def testgeneration_multiround(args,prompt,system_message=''):
    """generate test cases with multi-round conversation, each time generate one test case"""
    template_append="Generate another test method for the function under test. Your answer must be different from previously-generated test cases, and should cover different statements and branches."
    generated_tests=[]

    if args.model in models_nosys: #models don't support system message
        messages=[{"role": "user", "content": system_message+prompt}]
    else:
        messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    for i in range(args.num_tests):
        prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated=generator(prompt, 
                            max_new_tokens=args.max_tokens, 
                            temperature=args.temperature, 
                            return_full_text=False)
        
        generated_test=generated[0]['generated_text']
        print(generated_test)

        messages.append({"role": "assistant", "content": generated_test})
        messages.append({"role": "user", "content": template_append})

        generated_tests.append(generated_test)
    return generated_tests

# main loop
if __name__=='__main__':
    # original testeval code
    args=parse_args()
    os.environ["HUGGINGFACE_TOKEN"] = open("text_files/token.txt").read().strip()
    model_abbrv=args.model.split('/')[-1]

    print('Model:', args.model)
    output_dir = Path('predictions')

    dataset=read_jsonl('data/test_eval/leetcode-py.jsonl')
    system_template=open('test_eval/prompt/system.txt').read()
    system_message=system_template.format(lang='python')

    model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
    generator = pipeline("text-generation",model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map='auto', token=access_token)

    data_size=len(dataset)

    # new code:
    # if rag is enabled, setup the RAG pipeline
    if args.rag:
        print("Using RAG for test generation...")
        # init pipeline
        rag_pipeline = setup_rag_pipeline(provider="openai")

        # load knowledge base
        docs = knowledge_base_loader.load_json_from_github(
            github_url="https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json",
            content_field="prompt",
            id_field="task_id",
            metadata_fields=["code", "test_list"],
        )

        # add knowledge base to the pipeline
        rag_pipeline.add_knowledge_base(docs)

        # original testeval code
        testing_results=[]
        for i in tqdm(range(data_size)):
            data=dataset[i]
            func_name=data['func_name']
            desc=data['description']
            code=data['python_solution']
            difficulty=data['difficulty']
            code_withlineno=add_lineno(code)
            target_lines=data['target_lines']

            # new code:
            # create a query object with the description and function name
            current_query = query(
                text=str(data),
                context_type="test_generation",
            )

            # use the RAG pipeline to retrieve relevant context documents
            query_embedding = rag_pipeline.embedding_model.encode([current_query.text])[0]
            retrieved_docs = rag_pipeline.vector_store.similarity_search(query_embedding, 5)

            # if retrieved_docs is not empty, concatenate their content
            if retrieved_docs:
                # concatenate context documents into a single text
                context_text = "\n\n".join([doc.content for doc in retrieved_docs])
            else: 
                # if no context documents, use empty string
                context_text = "No relevant context available." 

            # load the augmented testeval prompt template for test generation with rag
            prompt_template=open('test_eval/prompt/template_base_aug.txt').read()

            # original testeval code
            prompt=prompt_template.format(lang='python', program=code, description=desc, func_name=func_name, context=context_text)
            generated_tests=testgeneration_multiround(args,prompt,system_message)
                    
            testing_data={'task_num':data['task_num'],'task_title':data['task_title'],'func_name':func_name,'difficulty':difficulty,'code':code,'tests':generated_tests}
            testing_results.append(testing_data)
            print('<<<<----------------------------------------->>>>')
            write_jsonl(testing_results, output_dir / f'{datetime.today().strftime("%m-%d")}_rag_{args.model}.jsonl')
        write_jsonl(testing_results, output_dir / f'{datetime.today().strftime("%m-%d")}_rag_{args.model}.jsonl')


    # original testeval code
    else:
        prompt_template=open('test_eval/prompt/template_base.txt').read()
        testing_results=[]
        for i in tqdm(range(data_size)):
            data=dataset[i]
            func_name=data['func_name']
            desc=data['description']
            code=data['python_solution']
            difficulty=data['difficulty']
            code_withlineno=add_lineno(code)
            target_lines=data['target_lines']

            #generate test case
            prompt=prompt_template.format(lang='python', program=code, description=desc, func_name=func_name)
            generated_tests=testgeneration_multiround(args,prompt,system_message)
                    
            testing_data={'task_num':data['task_num'],'task_title':data['task_title'],'func_name':func_name,'difficulty':difficulty,'code':code,'tests':generated_tests}
            testing_results.append(testing_data)
            print('<<<<----------------------------------------->>>>')
            write_jsonl(testing_results, output_dir / f'{datetime.today().strftime("%m-%d")}_standard_{args.model}.jsonl')

        write_jsonl(testing_results, output_dir / f'{datetime.today().strftime("%m-%d")}_standard_{args.model}.jsonl')