#!/bin/bash

source startup.sh

SCRATCH_HOME="/mnt/scratch/home/ismayilz"
HOME="/home/ismayilz"
CONDA_ENV=dycoder
CONDA=/home/ismayilz/.conda/condabin/conda
PROJECT_HOME=$SCRATCH_HOME/project-dycoder

cd $PROJECT_HOME/dycoder
${CONDA} run -n ${CONDA_ENV} --live-stream "$@"
