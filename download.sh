#!/bin/bash
set -e

# NQ
echo Downloading NQ . . .
wget -P data/nq https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz

# SQUAD
echo Downloading SQUAD . . .
wget -P data/squad https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

# HotPotQA
echo Downloading HotPotQA . . .
wget -P data/hotpotqa http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

echo Done!
