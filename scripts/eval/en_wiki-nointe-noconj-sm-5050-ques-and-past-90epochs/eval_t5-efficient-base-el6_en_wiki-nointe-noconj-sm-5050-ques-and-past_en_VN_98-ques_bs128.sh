#!/bin/bash

#SBATCH --job-name=t5-efficient-base-el6-eval-tense-en_wiki-nointe-noconj-sm-5050-ques-and-past
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
	--model_name_or_path 'google/t5-efficient-base-el6' \
	--do_learning_curve \
	--validation_file data/en_VN_98-ques/en_VN_98-ques_test.json.gz \
	--output_dir outputs/en_wiki-nointe-noconj-sm-5050-ques-and-past-90epochs/t5-efficient-base-el6-finetuning-en_wiki-nointe-noconj-sm-5050-ques-and-past-bs128/ \
	--per_device_eval_batch_size 16 \
	--predict_with_generate