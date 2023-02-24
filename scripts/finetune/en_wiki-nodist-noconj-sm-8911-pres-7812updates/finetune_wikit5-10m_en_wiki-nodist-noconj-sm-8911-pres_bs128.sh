#!/bin/bash

#SBATCH --job-name=wikit5-10m-finetune-tense-en_wiki-nodist-noconj-sm-8911-pres
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --time=02-00:00:00
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load CUDA
module load cuDNN
module load miniconda

source activate /gpfs/gibbs/project/frank/ref4/conda_envs/py38-agratt

python core/run_seq2seq.py \
	--model_name_or_path 'mueller-t5-checkpoints/wikit5-10m' \
	--do_train \
	--train_file data/en_wiki-nodist-noconj-sm-8911-pres/en_wiki-nodist-noconj-sm-8911-pres_train.json.gz \
	--output_dir outputs/en_wiki-nodist-noconj-sm-8911-pres-7812updates/wikit5-10m-finetuning-en_wiki-nodist-noconj-sm-8911-pres-bs128/ \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 32 \
	--overwrite_output_dir \
	--max_steps 7812