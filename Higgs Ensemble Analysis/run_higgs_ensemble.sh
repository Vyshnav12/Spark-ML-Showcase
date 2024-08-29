#!/bin/bash
#$ -l h_rt=6:00:00  # time needed in hours:mins:secs
#$ -pe smp 8 # number of cores requested (updated to match local[8] in Spark)
#$ -l rmem=90G # size of memory requested
#$ -o ../Output/higgs_ensemble_analysis_output.txt  # Updated output file name
#$ -j y # normal and error outputs into a single file (the file above)
#$ -cwd # run job from current directory

source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 90g --master local[8] higgs_ensemble_analysis.py
