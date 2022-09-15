#!/bin/bash

#SBATCH --job-name=clear_dataset_cache
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:01:00

rm -r ~/.cache/huggingface/datasets/json/
rm -r ~/.cache/huggingface/datasets/downloads/
rm ~/.cache/huggingface/datasets/*.lock
rm ~/.cache/huggingface/datasets/*.py
rm ~/.cache/huggingface/datasets/*.json
