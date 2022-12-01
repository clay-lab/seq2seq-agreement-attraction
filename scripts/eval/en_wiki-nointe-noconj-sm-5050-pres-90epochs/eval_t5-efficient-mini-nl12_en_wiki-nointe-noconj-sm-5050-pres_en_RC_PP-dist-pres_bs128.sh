#!/bin/bash

#SBATCH --job-name=t5-efficient-mini-nl12-eval-tense-en_wiki-nointe-noconj-sm-5050-pres
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
	--model_name_or_path 'google/t5-efficient-mini-nl12' \
	--do_learning_curve \
	--task translation_src_to_tgt \
	--train_file data/en_wiki-nointe-noconj-sm-5050-pres/en_wiki-nointe-noconj-sm-5050-pres_train.json.gz \
	--validation_file data/en_RC_PP-dist-pres/en_RC_PP-dist-pres_test.json.gz \
	--output_dir outputs/en_wiki-nointe-noconj-sm-5050-pres-90epochs/t5-efficient-mini-nl12-finetuning-en_wiki-nointe-noconj-sm-5050-pres-bs128/ \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=16 \
	--overwrite_output_dir \
	--predict_with_generate \
	--num_train_epochs 90.0