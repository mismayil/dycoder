#!/bin/bash

source startup.sh

SCRATCH_HOME="/mnt/scratch/home/ismayilz"
HOME="/home/ismayilz"
CONDA_ENV=coconut
CONDA=/home/ismayilz/.conda/condabin/conda
PROJECT_HOME=$SCRATCH_HOME/project-coconut

cd $PROJECT_HOME/coconut
${CONDA} run -n ${CONDA_ENV} --live-stream bash "$@"
