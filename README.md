## Abstract

In this repository, we propose a novel approach that utilizes Large Language Models (LLMs) for code inspection and test case generation, easing the burden of traditional methods. Additionally, to address the hallucination problem&mdash;in which LLMs produce incorrect outputs&mdash;we will implement a Retrieval Augmented Generation (RAG) pipeline to integrate supplementary knowledge sources

## Requirements
```
huggingface_hub==0.32.4
numpy==2.3.1
ollama==0.5.1
openai==1.97.0
Requests==2.32.4
sentence_transformers==4.1.0
sortedcontainers
torch==2.7.1
tqdm==4.64.0
transformers==4.53.2
```

## Setup


Install requirements (see above)

```bash
pip install -r requirements.txt
```

Create folders
```bash
mkdir predictions
mkdir results
mdkir text_files
```

Configure model access

```bash
echo "your openai key here" > text_files/key.txt
echo "your huggingface token here" > text_files/token.txt
```

Load TestEval data
```bash
git clone https://github.com/LLM4SoftwareTesting/TestEval.git test_eval
# move files to final location
mv template_base_aug.txt test_eval/prompt
mv test_eval/.coveragerc . 
mv test_eval/data_utils.py .
```

Cleanup files
```bash
cd test_eval
find . -type f ! \( -name 'eval_overall.py' -o -name 'format.py' -o -name 'system.txt' -o -name 'template_base.txt' -o -name 'template_base_aug.txt' \) -delete
find . -type d ! -path './prompt*' -delete
```

## File tree
Your file tree should now match the outline below. 

Files labeled "from testeval" have been downloaded directly from the [TestEval repository](https://github.com/LLM4SoftwareTesting/TestEval) 
```md
.
├── .coveragerc (from testeval)
├── data
│   ├── code_stack
│   │   └── datasets
│   │       └── bug_in_the_code_stack_alpaca_dataset.csv
│   ├── csv_to_json
│   │   └── converted.json
│   └── test_eval
│       ├── leetcode-py-all.jsonl
│       ├── leetcode-py-instrumented.jsonl
│       ├── leetcode-py.jsonl
│       └── tgt_paths.jsonl
├── data_utils.py (from testeval)
├── eval_review.py
├── main.py
├── predictions
├── README.md
├── requirements.txt
├── results
├── review_base.py
├── test_eval
│   └── prompt 
│       └── system.txt (from testeval)
│       └── template_base_aug.txt (from testeval)
│       └── template_base.txt (from testeval)
│   └── eval_overall.py (from testeval)
│   └── format.py (from testeval)
├── test_gen_hf.py
├── test_gen_openai.py
├── text_files
│   ├── key.txt
│   └── token.txt

```

## Experiments

Experimental configuration
``` md
temperature = 0
max tokens = 256
num tests = 20
```

### Run experiments: Automated Code Inspection
```bash
# generate code review
python review_base.py --rag {True/False} --model {model_name}
# evaluate correctness metrics
python eval_review.py --path {path_to_code_review} --output_file {path_to_output}
```

### Run experiments: Automated Test Case Generation
Base commands from Testeval
```bash
# generate raw test cases
python test_gen_{openai/hf}.py --model {model_name} --num_tests {num_tests} --rag {True/False}
# reformat test cases
python test_eval/format.py --mode overall --path {path_to_generated_tests}
# evaluate correctness and coverage metrics
python test_eval/eval_overall.py --path {path_to_formatted_generated_tests}
```
