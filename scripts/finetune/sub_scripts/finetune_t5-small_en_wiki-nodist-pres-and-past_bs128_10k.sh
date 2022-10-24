#!/bin/bash

#SBATCH --job-name=t5-small-finetune-tense-en_wiki-nodist-pres-and-past_10k
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
	--model_name_or_path 't5-small' \
	--do_train \
	--task translation_src_to_tgt \
	--train_file data/en_wiki-nodist-pres-and-past/sub_datasets/en_wiki-nodist-pres-and-past_train_10k.json \
	--validation_file data/en_wiki-nodist-pres-and-past/sub_datasets/en_wiki-nodist-pres-and-past_dev_100.json \
	--output_dir outputs/t5-small-finetuning-en_wiki-nodist-pres-and-past_10k-bs128/ \
	--per_device_train_batch_size=4 \
	--gradient_accumulation_steps=32 \
	--per_device_eval_batch_size=16 \
	--overwrite_output_dir \
	--predict_with_generate \
	--num_train_epochs 10.0
