# import necessary libraries

# for path
from pathlib import Path

# for progress bar
from tqdm import tqdm

# for environment variables
import os

# for using openai models
import openai
from openai import OpenAI

# for argument parsing
from argparse import ArgumentParser

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
openai.api_key=open("text_files/key.txt").read().strip()
client=OpenAI(api_key=openai.api_key)

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
    parser.add_argument("--lang", type=str, default='python')
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1'])
    parser.add_argument("--num_tests", type=int, default=10, help='number of tests generated per program')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=256)
    # new token for rag config
    parser.add_argument("--rag", type=bool, default=False, help='use RAG or not')
    return parser.parse_args()

# original testeval functions
# func to generate a single completion
def generate_completion(args,prompt,system_message=''):
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    code_output=response.choices[0].message.content
    return code_output

# func to generate test cases with multi-round conversation
def testgeneration_multiround(args,prompt,system_message=''):
    """generate test cases with multi-round conversation, each time generate one test case"""
    template_append="Generate another test method for the function under test. Your answer must be different from previously-generated test cases, and should cover different statements and branches."
    generated_tests=[]
    messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    for i in range(args.num_tests):
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        generated_test=response.choices[0].message.content
        messages.append({"role": "assistant", "content": generated_test})
        messages.append({"role": "user", "content": template_append})

        generated_tests.append(generated_test)
        print(generated_test)

    return generated_tests

# language extensions for testeval
lang_exts={'python':'py', 'java':'java', 'c++':'cpp'}

# main loop
if __name__=='__main__':
    # original testeval code
    args=parse_args()
    os.environ["OPENAI_API_KEY"] = open("text_files/key.txt").read().strip()

    print('Model:', args.model)
    output_dir = Path('predictions')

    dataset=read_jsonl('data/test_eval/leetcode-py.jsonl')
    system_template=open('test_eval/prompt/system.txt').read()
    system_message=system_template.format(lang='python')

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
            write_jsonl(testing_results, output_dir / f'{datetime.today().strftime("%m_%d")}_rag_{args.model}.jsonl')


        write_jsonl(testing_results, output_dir / f'{datetime.today().strftime("%m_%d")}_rag_{args.model}.jsonl')



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
            write_jsonl(testing_results, output_dir / f'{datetime.today().strftime("%m_%d")}_standard_{args.model}.jsonl')

        write_jsonl(testing_results, output_dir / f'{datetime.today().strftime("%m_%d")}_standard_{args.model}.jsonl')