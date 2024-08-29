#!/bin/bash
#$ -P rse-com6012
#$ -l h_rt=02:00:00  
#$ -pe smp 6
#$ -l rmem=12G
#$ -o ./Output/ALS_Clustering_Analysis_output.txt  
#$ -j y # Merge standard and error outputs into a single file (the file above)
#$ -cwd 

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 12g --executor-memory 12g --master local[6] ./als_clustering_analysis.py
