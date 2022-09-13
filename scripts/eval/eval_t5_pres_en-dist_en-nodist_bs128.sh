#!/bin/bash

#SBATCH --job-name=T5-base-eval-tense-en-dist-en-nodist
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
	--model_name_or_path 't5-base' \
	--do_learning_curve \
	--task translation_src_to_tgt \
	--train_file data/pres_en-dist_en-nodist/pres_en-dist_en-nodist_train.json.gz \
	--validation_file data/pres_en-nodist/pres_en-nodist_test.json.gz \
	--output_dir outputs/t5-finetuning-pres-en-dist-en-nodist-bs128/ \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=16 \
	--overwrite_output_dir \
	--predict_with_generate \