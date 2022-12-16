#!/bin/bash

#SBATCH --job-name=simplewiki-100m-eval-tense-en_wiki-nointe-noconj-sm-5050-pres
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
	--model_name_or_path 'mueller-t5-checkpoints/simplewiki-100m' \
	--do_learning_curve \
	--validation_file data/en_RC_PP_gen-dist-pres/en_RC_PP_gen-dist-pres_test.json.gz \
	--output_dir outputs/en_wiki-nointe-noconj-sm-5050-pres-7812updates/simplewiki-100m-finetuning-en_wiki-nointe-noconj-sm-5050-pres-bs128/ \
	--per_device_eval_batch_size 16 \
	--predict_with_generate