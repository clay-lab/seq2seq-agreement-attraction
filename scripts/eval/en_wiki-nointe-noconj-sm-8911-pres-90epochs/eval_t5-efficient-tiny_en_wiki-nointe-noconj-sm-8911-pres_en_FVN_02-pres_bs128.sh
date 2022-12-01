#!/bin/bash

#SBATCH --job-name=t5-efficient-tiny-eval-tense-en_wiki-nointe-noconj-sm-8911-pres
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
	--do_learning_curve \
	--task translation_src_to_tgt \
	--train_file data/en_wiki-nointe-noconj-sm-8911-pres/en_wiki-nointe-noconj-sm-8911-pres_train.json.gz \
	--validation_file data/en_FVN_02-pres/en_FVN_02-pres_test.json.gz \
	--output_dir outputs/en_wiki-nointe-noconj-sm-8911-pres-90epochs/t5-efficient-tiny-finetuning-en_wiki-nointe-noconj-sm-8911-pres-bs128/ \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=16 \
	--overwrite_output_dir \
	--predict_with_generate \
	--num_train_epochs 90.0