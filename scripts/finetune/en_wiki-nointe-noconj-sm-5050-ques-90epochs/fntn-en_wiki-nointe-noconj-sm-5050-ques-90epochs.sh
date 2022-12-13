#!/bin/bash
#SBATCH --output joblogs/fntn-en_wiki-nointe-noconj-sm-5050-ques-90epochs-%A_%a.txt
#SBATCH --array 0-15%2000
#SBATCH --job-name fntn-en_wiki-nointe-noconj-sm-5050-ques-90epochs
#SBATCH --nodes 1 --cpus-per-task 1 --mem 30GB --time 10:00:00 --gpus v100:1 --partition gpu --mail-type END,FAIL,INVALID_DEPEND

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/gibbs/project/frank/ref4/shared/seq2seq-agreement-attraction/scripts/finetune/en_wiki-nointe-noconj-sm-5050-ques-90epochs/fntn-en_wiki-nointe-noconj-sm-5050-ques-90epochs.txt --status-dir /gpfs/gibbs/project/frank/ref4/shared/seq2seq-agreement-attraction/joblogs

