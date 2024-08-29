#!/bin/bash
#$ -l h_rt=6:00:00  # time needed in hours:mins:secs
#$ -pe smp 2 # number of cores requested
#$ -l rmem=10G # size of memory requested 
#$ -o ../Output/log_mining_analysis_output.txt 
#$ -j y # normal and error outputs into a single file (the file above)
#$ -cwd # run job from current directory

source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 10g --master local[2] log_mining_analysis.py 
