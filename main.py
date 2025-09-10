# import necessary libraries

# for environment variables
import os

# for path
from pathlib import Path

# for type annotations
from typing import List, Dict, Any, Optional, Callable

# for time tracking
import time
from tqdm import tqdm

# for data classes
from dataclasses import dataclass

# for abstract base classes
from abc import ABC, abstractmethod

# for making HTTP requests
import requests

# for sentence transformers
from sentence_transformers import SentenceTransformer

# for using openAI models
from openai import OpenAI

# for handling JSON data
import json

# for handling csv data
import csv

# for numerical operations and similarity calculations
import numpy as np
from numpy.linalg import norm

# for random selection
import random

# for prompt formatting
import textwrap

# for interface implementation
from huggingface_hub import InferenceClient

# for local implementation
import ollama

# core data structures

# progress tracking callback type
progress_callback = Callable[[str, float], None]

# represents a document in the knowledge base
@dataclass
class document:
    # unique identifier for the document
    id: str

    # content of the document
    content: str

    # metadata associated with the document
    metadata: Dict[str, Any]

    # embedding vector for the document
    # currently not configured, will be added later
    embedding: Optional[List[float]] = None

# represents a query to the RAG system
@dataclass
class query:
    # query text
    text: str

    # context type for the query
    # will allow 'code_inspection' or 'test_generation'
    context_type: str 

    # additional metadata for the query
    metadata: Dict[str, Any] = None

# represents a response from the RAG system
@dataclass
class RAG_response:
    # generated answer from the LLM
    answer: str

    # list of retrieved documents relevant to the query
    retrieved_docs: List[document]
 
# testeval task
@dataclass
class test_eval_task:
    task_num: str
    task_title: str
    difficulty: str
    func_name: str
    description: str
    python_solution: str
    blocks: List[str]

# code stack task
@dataclass
class code_stack_task:
    output: str
    output_missing_colon: str
    bug_line_number_missing_colon: str
    output_missing_parenthesis: str
    bug_line_number_missing_parenthesis: str
    output_missing_quotation: str
    bug_line_number_missing_quotation: str
    output_missing_comma: str
    bug_line_number_missing_comma: str
    output_mismatched_quotation: str
    bug_line_number_mismatched_quotation: str
    output_mismatched_bracket: str
    bug_line_number_mismatched_bracket: str
    output_keywords_as_identifier: str
    bug_line_number_keywords_as_identifier: str

# helper functions

# helper function for json reading
def read_json(path):
    data = []
    with open(path,'r') as file:
        data = json.load(file)
        json_list = list(data)
    file.close()
    return json_list

# helper function for jsonl reading
def read_jsonl(path):
    data_list = []
    with open(path,'r') as file:
        for line in file:
            data_list.append(json.loads(line.strip()))
    file.close()
    return data_list

# helper function for json writing
def write_jsonl(data, path):
    with open(path,'w') as file:
        for datum in data:
            file.write(json.dumps(datum)+'\n')
    file.close()

# helper function to convert csv file to json
def csv_to_json(csv_filepath, json_filepath):
    data = []
    with open(csv_filepath, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data.append(row)

    csvfile.close()

    with open(json_filepath, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)

    jsonfile.close()

# helper function to save results to a JSONL file
def save_to_jsonl(testing_data: Dict[str, Any], output_file: str, progress_callback: Optional[progress_callback] = None) -> None:
    # append testing data to the JSONL file
    try:
        # progress tracking
        if progress_callback:
            progress_callback(f"Saving results to {output_file}", 0.0)

        # open file in append mode
        with open(output_file, 'a') as file:
            # add testing data as a JSON string
            file.write(json.dumps(testing_data) + "\n")

            # progress tracking
            if progress_callback:
                progress_callback(f"Saved entry to {output_file}", 0.0) 

    # exception handling
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

    # close file
    file.close()

# helper function to separate data
def select_output_code_stack(task: code_stack_task):
    # create empty list
    outputs = []

    # get all output attributes and save as key value pairs
    output_attrs = {attr: getattr(task, attr) for attr in vars(task) if attr.startswith('output')}

    # remove key value pairs that have empty values
    output_attrs = {k: v for k, v in output_attrs.items() if v.strip()}

    # if there are outputs
    if output_attrs:
        # randomly select one of the outputs and save the key and value
        key, value = random.choice(list(output_attrs.items()))
        return key, value

    # otherwise, if there are no outputs
    else:        
        # error
        return None, None

# dataset loading functions

# load testeval data
def load_test_eval_dataset(folder_path: str, progress_callback: Optional[progress_callback] = None) -> Dict[str, test_eval_task]:
    # progress tracking
    if progress_callback:
        progress_callback("Loading dataset", 0.0)
    
    # load tasks from files in the dataset folder
    # currently hardcoded for json files
    path = Path(folder_path)
    task_files = list(path.glob("*all.json*"))

    # create a list to store tasks
    tasks = []

    # progress tracking
    if progress_callback:
        progress_callback(f"Getting dataset from path: {folder_path}", 0.0)
    
    # for every file pulled
    for i, task_file in enumerate(task_files):
        try:
            # read file
            with open(task_file) as json_file:
                json_list = list(json_file)
                for json_str in json_list:
                    # add each json string to task_data
                    task_data = json.loads(json_str)

                    task = test_eval_task(
                        task_num=task_data.get('task_num', ''),
                        task_title=task_data.get('task_title', ''),
                        difficulty=task_data.get('difficulty', ''),
                        func_name=task_data.get('func_name', ''),
                        description=task_data.get('description', ''),
                        python_solution=task_data.get('python_solution', ''),
                        blocks=task_data.get('blocks', [])
                    )
                    
                    # save task
                    tasks.append(task)

        # exception handling    
        except Exception as e:
            print(f"Error loading task from {task_file}: {e}")
    
    # progress tracking
    if progress_callback:
        progress_callback(f"Loaded {len(tasks)} tasks", 1.0)
    
    # close file
    json_file.close()

    # return tasks
    return tasks

# load codestack data
def load_code_stack_dataset(folder_path: str, progress_callback: Optional[progress_callback] = None) -> Dict[str, code_stack_task]:
    # progress tracking
    if progress_callback:
        progress_callback("Loading dataset", 0.0)
    
    # load tasks from files in the dataset folder
    # currently hardcoded for json files
    path = Path(folder_path)
    task_files = list(path.glob("*.json*"))

    # create a list to store tasks
    tasks = []

    # progress tracking
    if progress_callback:
        progress_callback(f"Getting dataset from path: {folder_path}", 0.0)
    
    # for every file pulled
    for i, task_file in enumerate(task_files):
        try:
            # read file
            with open(task_file) as json_file:
                data = json.load(json_file)
                json_list = list(data)

                for task_data in json_list:
                    # add each json string to task_data
                    task = code_stack_task(
                        output=task_data.get('output',''),
                        output_missing_colon=task_data.get('output_missing_colon',''),
                        bug_line_number_missing_colon=task_data.get('bug_line_number_missing_colon',''),
                        output_missing_parenthesis=task_data.get('output_missing_parenthesis',''),
                        bug_line_number_missing_parenthesis=task_data.get('bug_line_number_missing_parenthesis',''),
                        output_missing_quotation=task_data.get('output_missing_quotation',''),
                        bug_line_number_missing_quotation=task_data.get('bug_line_number_missing_quotation',''),
                        output_missing_comma=task_data.get('output_missing_comma',''),
                        bug_line_number_missing_comma=task_data.get('bug_line_number_missing_comma',''),
                        output_mismatched_quotation=task_data.get('output_mismatched_quotation',''),
                        bug_line_number_mismatched_quotation=task_data.get('bug_line_number_mismatched_quotation',''),
                        output_mismatched_bracket=task_data.get('output_mismatched_bracket',''),
                        bug_line_number_mismatched_bracket=task_data.get('bug_line_number_mismatched_bracket',''),
                        output_keywords_as_identifier=task_data.get('output_keywords_as_identifier',''),
                        bug_line_number_keywords_as_identifier=task_data.get('bug_line_number_keywords_as_identifier','')
                    )
                    
                    # save task
                    tasks.append(task)

        # exception handling    
        except Exception as e:
            print(f"Error loading task from {task_file}: {e}")
    
    # progress tracking
    if progress_callback:
        progress_callback(f"Loaded {len(tasks)} tasks", 1.0)
    
    # close file
    json_file.close()

    # return tasks
    return tasks

# prompt generation

# function to generate prompt based on query type and context documents
def generate_base_prompt(query: query, context_docs: Optional[List[document]] = None) -> str:
    # check if context documents are available
    if context_docs:
        # concatenate context documents into a single text
        context_text = "\n\n".join([doc.content for doc in context_docs])
    else: 
        # if no context documents, use empty string
        context_text = "No relevant context available."       

    # initialize default values
    func_name = 'unknown_function'
    description = 'No description provided.'
    program = []

    # if query context type is test generation
    if query.context_type == "test_generation":
        # parse the test_eval_task object from query text
        task_obj = eval(query.text)
        if isinstance(task_obj, test_eval_task):
            func_name = task_obj.func_name
            description = task_obj.description
            program = task_obj.blocks

        # if context type is test generation, format prompt accordingly 
        prompt = f"""
        Please write a test method for the function '{func_name}' given the following program under test, 
        the function description, and the knowledge base {context_text}. 
        Your answer should only contain one test input.

        Program under test:
        ----
        {program}
        ----

        Function description for '{func_name}':
        ----
        {description}
        ----

        Your test method should begin with:
        def test_{func_name}():
            solution=Solution()

        You should only generate the test case, without any additional explanation. 
        """     
    elif query.context_type == "bug_one_line":
        prompt = f"""
        You are an expert software engineer performing code inspection. 
        Use the following knowledge base to help identify potential issues or bugs in the given code.

        Knowledge Base:
        {context_text}

        Code to inspect:
        {query.text}

        Please a one line code review identifying the type of defect in the code. 
        If there are no visible bugs, please state "Code is visibly bug-free"

        Review:
        """   
    elif query.context_type == "bug_multi_line":
        prompt = f"""
        You are an expert software engineer performing code inspection. 
        Use the following knowledge base to help identify potential issues or bugs in the given code.

        Knowledge Base:
        {context_text}

        Code to inspect:
        {query.text}

        Please provide a detailed code review identifying:
        1. Potential bugs or errors
        2. Code quality issues
        3. Best practice violations
        4. Security concerns (if any)

        Review:
        """      
    else:
        prompt = f"""
        Context: {context_text}

        Query: {query.text}

        Response:
        """

    # return generated prompt, with unnecessary whitespace removed
    return textwrap.dedent(prompt)


# abstract base classes

# abstract base class for embedding models
class embedding_model(ABC):
    # abstract method for encoding texts into embeddings
    # currently not configured, will be added later
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass

# abstract base classes for vector storage
class vector_storage(ABC):
    # abstract method for adding documents
    # currently not configured, will be added later
    @abstractmethod
    def add_documents(self, documents: List[document]) -> None:
        pass
    
    # abstract method for retrieving similar documents based on query embedding
    # currently not configured, will be added later
    @abstractmethod
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[document]:
        pass

# abstract base class for LLM interaction
class LLM_interface(ABC):
    # abstract method for generating responses with retrieved context
    # currently not configured, will be added later
    @abstractmethod
    def generate_response(prompt: str, model: str, context: Optional[List[document]] = None, progress_callback: Optional[progress_callback] = None) -> str:
        if model == "openai":
            open_AI.generate_response(prompt, context, progress_callback)
        elif model == "hf":
            hf.generate_response(prompt, context, progress_callback)
        else:
            print("Model not supported")

# concrete implementations

# LLM interface implementation hardcoded to use openai models
class open_AI(LLM_interface):
    # setup
    def __init__(self,
                 # current default model is gpt-4o
                 model: str = "gpt-4o",
                 
                 # optional API key, defaults to environment variable
                 api_key: Optional[str] = None):
        
        # initialize parameters
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    # function to generate response using OpenAI GPT
    def generate_response(self, prompt: str, context: Optional[List[document]] = None, progress_callback: Optional[progress_callback] = None) -> str:
        try:
            # progress tracking
            if progress_callback:
                progress_callback("Preparing prompt for OpenAi model", 0.2)

            # format prompt for OpenAI chat completion
            prompt_format=[
                    {"role": "system", "content": "You are an expert software engineer and tester. Provide detailed, accurate analysis based on the given context."},
                    {"role": "user", "content": prompt}
            ]

            # progress tracking
            if progress_callback:
                progress_callback("Sending request to OpenAi", 0.5)

            # generate response using OpenAI chat completions
            response = self.client.chat.completions.create(
                # use specified model
                model=self.model,

                # format messages for chat completion
                messages=prompt_format,
        
                # default parameters for generation
                temperature=0.3,
                max_tokens=2000
            )

            # progress tracking
            if progress_callback:
                progress_callback("Processing OpenAI response", 0.5)

            # check if response is valid
            if not response or not response.choices or len(response.choices) == 0:
                raise ValueError("No valid response from OpenAI API")

            # if response is valid, progress tracking
            else:
                if progress_callback:
                    progress_callback("Valid response received", 1.0)

            # return the generated content from the response
            return response.choices[0].message.content
        
        # exception handling
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response. Please check your OpenAI API key and try again."

# LLM interface implementation using ollama locally run models
class ollama_class(LLM_interface):
    # setup
    def __init__(self,
                 # current default model is tinyllama
                 model: str = "tinyllama"):
        
        # initialize parameters
        self.model = model

    def generate_response(self, prompt: str, context: Optional[List[document]] = None) -> str:
        # format prompt
        prompt_format=[
            {"role": "system", "content": "You are an expert software engineer and tester. Provide detailed, accurate analysis based on the given context."},
            {"role": "user", "content": prompt}
        ]

        # use the ollama chat function with one user message
        try:
            response = ollama.chat(
                model=self.model,
                messages=prompt_format,
                think=False
            )

            # return llm response
            return(response['message']['content'])

        # exception handling
        except Exception as e:
            print(f"Error generating response from ollama model: {e}")
            return "Error generating response."

# LLM interface implementation using InferenceClient with external provider
class hf(LLM_interface):
    def __init__(self, 
                 model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 hf_token: Optional[str] = None,
                 provider: Optional[str] = "nebius",
                 max_tokens: int = 2000,
                 temperature: float = 0.3,
                 timeout: int = 120):
        
        # save parameters
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # initialize hugging face InferenceClient with provider
        self.client = InferenceClient(
            model=model_name,
            token=self.hf_token,
            provider=provider
        )

    def generate_response(self, prompt: str, context: Optional[List[document]] = None) -> str:
        # format prompt
        prompt_format=[
            {"role": "system", "content": "You are an expert software engineer and tester. Provide detailed, accurate analysis based on the given context."},
            {"role": "user", "content": prompt}
        ]

        try:
            # use the chat completion endpoint with one user message
            response = self.client.chat.completions.create(
                messages=prompt_format,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # return llm response
            return response.choices[0].message.content.strip()
        
        # exception handling
        except Exception as e:
            print(f"Error generating response from Hugging Face model: {e}")
            return "Error generating response."


# sentence transformer embedding model
# currently configured to use all-MiniLM-L6-v2
# subject to change
class sentence_transformer_embedding(embedding_model):
    # implement embedding model using sentence transformers
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)
        # store model name for reference
        self.model_name = model_name
    
    # function to encode texts into embeddings
    def encode(self, texts: List[str], progress_callback: Optional[progress_callback] = None) -> List[List[float],]:
        # encode the input texts into embeddings
        try:
            # progress tracking
            if progress_callback:
                progress_callback("Generating sentence transformer embeddings", 0.0)
            
            # use sentence transformers to generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # progress tracking
            if progress_callback:
                progress_callback("Embeddings generated successfully", 1.0)

            # convert embeddings to list format
            return embeddings.tolist()
        
        # exception handling
        except Exception as e:
            # log error and return fallback dimensions
            print(f"Error generating sentence transformer embeddings: {e}")
            
            # fallback dimensions based on model
            dim = 384 if "MiniLM" in self.model_name else 768
            return [[0.0] * dim for _ in texts]

# in-memory vector storage implementation
# currently a placeholder
class in_memory_vector_storage(vector_storage):
    # setup in-memory vector store
    def __init__(self):
        self.documents: List[document] = []
    
    # function to add documents to the vector store
    def add_documents(self, documents: List[document]) -> None:
        # add documents 
        self.documents.extend(documents)
    
    # function to perform similarity search
    def similarity_search(self, query_embedding: List[float], n: int = 5) -> List[document]:
        # if no documents, return empty list
        if not self.documents:
            return []
        
        # calculate similarities for all documents
        similarities = []
        query_vec = np.array(query_embedding)
        
        # iterate through documents and calculate cosine similarity
        for doc in self.documents:
            # if document has embedding, calculate similarity
            if doc.embedding is not None:
                # calculate cosine similarity
                doc_vec = np.array(doc.embedding)
                similarity = np.dot(query_vec, doc_vec) / (norm(query_vec) * norm(doc_vec))
                similarities.append((doc, similarity))
            # otherwise, if no embedding is available
            else:
                # if no embedding, assign low similarity
                similarities.append((doc, 0.0))
        
        # sort by similarity (descending) and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in similarities[:n]]

# RAG pipeline 
# currently not configured for contextual retrieval
class RAG_pipeline:
    # setup RAG pipeline with embedding model, vector store, and LLM
    def __init__(self, 
                 embedding_model: embedding_model,
                 vector_store: vector_storage,
                 llm: LLM_interface):
        
        # store references to components
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.dataset = None
        self.knowledge_base_loaded = False
        self.tasks = {}

    # function to get code requirements from a file
    def get_requirements(self, file_path: str, progress_callback: Optional[progress_callback] = None):
        # progress tracking
        if progress_callback:
            progress_callback(f"Getting requirements from path: {file_path}", 0.0)

        # open file
        with open(file_path, 'r') as file:
            # read file contents
            requirements = file.read()

        # progress tracking
        if progress_callback:
            progress_callback(f"Successfully received requirements", 0.0)        

        # close file
        file.close()

        # return requirements as string
        return requirements
        
    # function to add knowledge base documents
    def add_knowledge_base(self, documents: List[Dict[str, Any]], progress_callback: Optional[progress_callback] = None) -> None:
        # progress tracking
        if progress_callback:
            progress_callback("Preparing documents for vector store", 0.0)
        
        # add documents to the vector store with embeddings
        doc_objects = []
        texts = [doc['content'] for doc in documents]

        # progress tracking
        if progress_callback:
            progress_callback("Generating embeddings for documents", 0.2)
        
        # encode texts into embeddings
        embeddings = self.embedding_model.encode(texts, progress_callback)

        # progress tracking
        if progress_callback:
            progress_callback("Creating document objects", 0.7)

        # iterate through documents and create document objects
        for i, doc in enumerate(documents):
            # create document object with id, content, metadata, and embedding
            doc_obj = document(
                id=doc.get('id', f"doc_{i}"),
                content=doc['content'],
                metadata=doc.get('metadata', {}),
                embedding=embeddings[i]
            )
            # append to the list of document objects
            doc_objects.append(doc_obj)
        
        # progress tracking
        if progress_callback:
            progress_callback("Adding documents to vector store", 0.9)

        # add documents to the vector store
        self.vector_store.add_documents(doc_objects)

        # progress tracking
        if progress_callback:
            progress_callback("Knowledge base successfully updated", 1.0)

        # mark knowledge base as loaded
        self.knowledge_base_loaded = True
        
        # log the number of documents added
        if progress_callback:
            progress_callback(f"Added {len(doc_objects)} documents to knowledge base", 1.0)
    
    # function to retrieve one response through the RAG pipeline
    def first_round_response(self, query: query, k: int = 5, progress_callback: Optional[progress_callback] = None) -> RAG_response:        
        # check if knowledge base is loaded
        if not self.knowledge_base_loaded:
            raise ValueError("Warning: No knowledge base loaded. Proceeding with empty context")

        # progress tracking
        if progress_callback:
            progress_callback("Starting query processing", 0.0)

        # encode query
        # progress tracking
        if progress_callback:
            progress_callback("Encoding query into embedding", 0.1)
            
        # encode the query text into an embedding
        query_embedding = self.embedding_model.encode([query.text], progress_callback)[0]
        
        # retrieve relevant documents
        # progress tracking
        if progress_callback:
            progress_callback("Retrieving relevant documents", 0.2)
        # perform similarity search in the vector store
        retrieved_docs = self.vector_store.similarity_search(query_embedding, k)
        
        # generate prompt
        # progress tracking
        if progress_callback:
            progress_callback("Generating prompt for LLM", 0.3)

        # generate appropriate prompt based on query type and retrieved documents
        prompt = generate_base_prompt(query, retrieved_docs)

        # generate response using LLM
        # progress tracking
        if progress_callback:
            progress_callback("Generating response with LLM", 0.4)       
        
        # generate response using the LLM with the prompt and retrieved documents
        response = self.llm.generate_response(prompt, retrieved_docs, progress_callback)

        # progress tracking
        if progress_callback:
            progress_callback("Query processing completed", 1.0)
        
        # return RAG response with answer, retrieved documents, and metadata
        return RAG_response(
            answer=response,
            retrieved_docs=retrieved_docs,
        )


    # function to retrieve multiple responses through the RAG pipeline
    # and generate multiple test cases
    def multiple_test_gen(self, original_query: query, retrieved_docs: str, results: List[RAG_response], batch_size: int,
                         progress_callback: Optional[progress_callback] = None) -> List[RAG_response]:
        
        # save original response as the first result
        results.append(results[0])

        # run the loop for batch size
        for i in range(batch_size):

            # update prompt for each cycle based on previously generated results
            prompt = f"""
                Generate another test method for the following query, {original_query.text}.
                Your answer must be different from previously-generated test cases, {results}. 
                This response should cover different statements and branches.
                You should only generate one test case, without any additional explanation,
                with the same function name and format as previously generated test cases.
            """

            # generate response using the RAG pipeline
            response = self.llm.generate_response(prompt, retrieved_docs, progress_callback)
            
            # append the response to the queries list
            results.append(response)

            # printing to console
            if isinstance(response, str):
                print(response.strip())
            elif hasattr(response, "answer"):
                print(response.answer.strip())
            else:
                print(str(response).strip())

        # convert the original query text to a test_eval_task object
        task_obj = eval(original_query.text)
        if isinstance(task_obj, test_eval_task):
            task_num = task_obj.task_num
            task_title = task_obj.task_title
            func_name = task_obj.func_name
            difficulty = task_obj.difficulty
            code = task_obj.python_solution
        
        # save the testing data
        testing_data={'task_num':task_num,'task_title':task_title,'func_name':func_name,'difficulty':difficulty,'code':code,'tests':results}

        # return the testing data
        return testing_data

# class for loading knowledge base documents
class knowledge_base_loader:
    @staticmethod
    # function to load json data from a github repo
    def load_json_from_github(github_url: str,
                              content_field: str,
                              id_field: str = None,
                              metadata_fields: List[str] = None,
                              progress_callback: Optional[progress_callback] = None,
                              is_jsonl: bool = False) -> List[Dict[str, Any]]:

        try:
            # progress tracking
            if progress_callback:
                progress_callback("Converting GitHub URL to raw format", 0.1)
            
            # if necessary, convert github url to raw content url
            if "github.com" in github_url and "/blob/" in github_url:
                raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            else:
                raw_url = github_url
            
            # progress tracking
            if progress_callback:
                progress_callback("Downloading file from GitHub", 0.3)
            
            # save the file
            response = requests.get(raw_url, timeout=30)

            # check if the request was successful
            response.raise_for_status()
            
            # progress tracking
            if progress_callback:
                progress_callback("Parsing data", 0.6)
            
            # handle jsonl and json
            if is_jsonl or github_url.endswith('.jsonl'):
                # parse jsonl
                data = []
                for line in response.text.strip().split('\n'):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON line: {e}")
                            continue
            else:
                # parse regular json
                data = response.json()
                
                # handle both single object and array of objects
                if isinstance(data, dict):
                    data = [data]
                elif not isinstance(data, list):
                    raise ValueError("JSON must contain an object or array of objects")
            
            # progress tracking
            if progress_callback:
                progress_callback("Processing documents", 0.8)
            
            # initialize documents list
            documents = []
            metadata_fields = metadata_fields or []
            
            # iterate through each item in the data
            for i, item in enumerate(data):
                # skip if item is not a dictionary
                if not isinstance(item, dict):
                    continue
                
                # extract content
                content = item.get(content_field, "")
                # if content is none or empty, skip this item
                if not content:
                    print(f"Warning: No content found for item {i} using field '{content_field}'")
                    continue
                
                # generate or extract ID
                doc_id = item.get(id_field) if id_field else f"github_doc_{i}"
                
                # extract metadata
                metadata = {
                    "source": "github",
                    "github_url": github_url,
                    "raw_url": raw_url
                }
                
                # add metadata fields if they exist in the item
                for field in metadata_fields:
                    if field in item:
                        metadata[field] = item[field]
                
                # add any remaining fields as metadata (excluding content and id fields)
                for key, value in item.items():
                    if key not in [content_field, id_field] and key not in metadata_fields:
                        metadata[key] = value
                
                # add document to the list
                documents.append({
                    "id": str(doc_id),
                    "content": str(content),
                    "metadata": metadata
                })
            
            # progress tracking
            if progress_callback:
                progress_callback("Loading completed", 1.0)
            
            if progress_callback:
                progress_callback(f"Loaded {len(documents)} documents from GitHub: {github_url}", 1.0)

            return documents
            
        except requests.RequestException as e:
            print(f"Error downloading from GitHub: {e}")
            return []
        
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON from GitHub URL: {e}")
            return []
        
        except Exception as e:
            print(f"Error loading JSON from GitHub: {e}")
            return []

# function to setup RAG pipeline with specified provider and dataset
def setup_rag_pipeline(provider: str, hf_provider: str, progress_callback: Optional[progress_callback] = None) -> RAG_pipeline:

    # setup RAG pipeline based on provider
    # currently supports "hf" for hugging face models and "openai" for openai models 
    if provider == "hf":
        # print provider information
        # hardcoded
        print(f"Using Hugging Face via provider '{hf_provider}'...\n")

        # default configuration for hugging face models
        default_hf_config = {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "hf_token": os.getenv("HUGGINGFACE_TOKEN"),
            "provider": hf_provider,
            "max_retries": 3,
            "retry_delay": 5,
        }
        
        # initialize components
        # embedding model using sentence transformers
        embedding_model = sentence_transformer_embedding()
        llm = hf(**default_hf_config)
    
    elif provider == "ollama":
        # print provider information
        # hardcoded
        print(f"Using ollama'...\n")

        default_ollama_config = {
            "model": "tinyllama"
        }

        # initialize components
        # embedding model using sentence transformers
        embedding_model = sentence_transformer_embedding()
        llm = ollama(**default_ollama_config)


    elif provider == "openai":
        # set environment variable to disable parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # print provider information
        # hardcoded
        print("Using OpenAI models...\n")

        # default configuration for OpenAI models
        default_openai_config = {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        
        # initialize components
        # embedding model using sentence transformers
        embedding_model = sentence_transformer_embedding()
        llm = open_AI(**default_openai_config)

    # if provider is not supported, raise error
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # initialize in-memory vector store
    vector_store = in_memory_vector_storage()

    # create RAG pipeline with components
    pipeline = RAG_pipeline(embedding_model, vector_store, llm)

    # return the configured RAG pipeline
    return pipeline

# simple progress tracker to monitor stages and elapsed time
class progress_tracker:
    # setup
    def __init__(self):
        self.start_time = time.time()
        self.stages = []
    
    # function to track progress and log messages
    def callback(self, message: str, progress: float):
        # get current time
        current_time = time.time()

        # calculate elapsed time
        elapsed = current_time - self.start_time
        
        # track the stage with message, progress, and elapsed time
        self.stages.append({
            'message': message,
            'progress': progress,
            'elapsed': elapsed
        })
        
        # display real-time progress
        if progress > 0:
            # estimate total time
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            # print progress message with estimated remaining time
            print(f"{message} ({progress:.1%}) - Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s\n")
        
        # otherwise, if no progress is made, just print the message
        else:
            print(f"{message} - Elapsed: {elapsed:.1f}s\n")