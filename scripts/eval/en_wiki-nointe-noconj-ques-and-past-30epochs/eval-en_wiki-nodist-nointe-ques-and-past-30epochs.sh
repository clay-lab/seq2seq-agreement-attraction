#!/bin/bash
#SBATCH --output joblogs/eval-en_wiki-nodist-nointe-ques-and-past-30epochs-%A_%a.txt
#SBATCH --array 0-3%2000
#SBATCH --job-name eval-en_wiki-nodist-nointe-ques-and-past-30epochs
#SBATCH --nodes 1 --cpus-per-task 1 --mem 30GB --time 10:00:00 --gpus v100:1 --partition gpu --mail-type END,FAIL,INVALID_DEPEND --dependency=afterok:

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/gibbs/project/frank/ref4/shared/seq2seq-agreement-attraction/scripts/eval/en_wiki-nointe-noconj-ques-and-past-30epochs/eval-en_wiki-nodist-nointe-ques-and-past-30epochs.txt --status-dir /gpfs/gibbs/project/frank/ref4/shared/seq2seq-agreement-attraction/joblogs

