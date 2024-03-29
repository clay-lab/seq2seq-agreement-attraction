#!/bin/bash

#SBATCH --job-name=t5-efficient-mini-nl24-finetune-tense-en_wiki-nodist-ques-and-past
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --time=10:00:00
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load CUDA
module load cuDNN
module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/py38-agratt

python core/run_seq2seq.py \
	--model_name_or_path 'google/t5-efficient-mini-nl24' \
	--do_train \
	--train_file data/en_wiki-nodist-ques-and-past/en_wiki-nodist-ques-and-past_train.json.gz \
	--output_dir outputs/en_wiki-nodist-ques-and-past-30epochs/t5-efficient-mini-nl24-finetuning-en_wiki-nodist-ques-and-past-bs128/ \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 32 \
	--overwrite_output_dir \
	--num_train_epochs 30.0
