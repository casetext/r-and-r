# Long Context Needs Some R&R: Pushing Performance with Reprompting and In-Context Retrieval
Devanshu Agrawal, Shang Gao, and Martin Gajek

## Abstract

Long-context Large language models (LLMs) hold promise for tasks such as question-answering (QA) over long documents, 
but they tend to be biased towards the beginning and end of context documents-- a phenomenon termed the "lost in the middle" effect. 
Here, we introduce *R&R*---a combination of two novel prompt-based methods called *reprompting* and *in-context retrieval* (ICR)---to alleviate this effect in document-based QA. 
In reprompting, we repeat the instructions to answer the question periodically throughout the document nearly verbatim; 
in ICR, on the other hand, we instruct the LLM to retrieve the top *k* pages most relevant to the given question, which are then used as an abbreviated context for QA. 
In R&R, we run ICR with the retrieval instructions reprompted throughout the document. 
We test R&R with GPT-4 Turbo and Claude-2.1 on documents up to 80k tokens in length and find it is often able to boost QA accuracy significantly. 
We perform additional analysis to elucidate the mechanism driving R&R, including reducing the distance between relevant context and the instructions. 
Finally, we show that compared to short-context chunkwise methods, R&R enables the use of larger chunks that cost fewer LLM calls and output tokens, while minimizing the price in accuracy.


### Description

This repository includes code and scripts to reproduce the results presented in our paper of the same title.


## Requirements

- python >= 3.9 (along with standard Anaconda packages including numpy)
- anthropic
- dotenv
- openai
- tenacity
- tiktoken
- xopen


## Setup

### Environment variables

Create a directory named `configs` inside a clone of this repo. 
Then create a file named `configs/openai.env` with the following text:

```
MODEL_NAME=[model name]
OPENAI_ORG_ID=[your org ID]
OPENAI_API_KEY=[your OpenAI API key]
```

where the right-hand sides are replaced with your respective values. 
Here `model name` could be e.g., `gpt-4-1106-preview`. 
Similarly, create a file named `configs/anthropic.env` with analogous variables for Anthropic in place of OpenAI.


### Datasets

To download the NQ, SQuAD, and HotPotQA datasets, run the following:

    bash download.sh

To generate the PubMed dataset, run the following commands:

    cd ..
    git clone https://github.com/dagrawa2/pubmed
    cd pubmed
    python download.py
    python unpack.py
    python parse.py
    cd ../r-and-r
    python build_pubmed.py

Note some of the scripts may run for hours, 
and `build_pubmed.py` uses the OpenAI model specified in the `configs` directory to generate the PubMed dataset.


## Running experiments

All experiments call the `run` function located in [src/run.py](src/run.py), 
which is the main function of the code.

For the main experiments of the paper, run the following:

    python baseline_vs_reprompt.py
    python cr_wwo_reprompt.py

Note these will run with the OpenAI model, 
but you can change to the Anthropic model by editing the above two scripts.

For additional analysis, run the following:

    python analysis.py

The above runs will populate the `results` directory.
