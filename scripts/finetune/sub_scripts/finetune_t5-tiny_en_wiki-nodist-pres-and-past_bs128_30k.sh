#!/bin/bash

#SBATCH --job-name=t5-tiny-finetune-tense-en_wiki-nodist-pres-and-past_30k
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
	--model_name_or_path 'google/t5-efficient-tiny' \
	--do_train \
	--task translation_src_to_tgt \
	--train_file data/en_wiki-nodist-pres-and-past/sub_datasets/en_wiki-nodist-pres-and-past_train_30k.json.gz \
	--validation_file data/en_wiki-nodist-pres-and-past/sub_datasets/en_wiki-nodist-pres-and-past_dev_300.json.gz \
	--output_dir outputs/t5-tiny-finetuning-en_wiki-nodist-pres-and-past_30k-bs128/ \
	--per_device_train_batch_size=4 \
	--gradient_accumulation_steps=32 \
	--per_device_eval_batch_size=16 \
	--overwrite_output_dir \
	--predict_with_generate \
	--num_train_epochs 10.0
