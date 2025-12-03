#!/bin/bash

SCRATCH_HOME="/mnt/scratch/home/ismayilz"
HOME="/home/ismayilz"
CONDA_ENV=coconut
CONDA=/home/ismayilz/.conda/condabin/conda
PROJECT_HOME=$SCRATCH_HOME/project-coconut

# symlink folders
rm -rf ~/.ssh
ln -s $SCRATCH_HOME/.ssh $HOME/.ssh
rm -rf ~/.cache
ln -s $SCRATCH_HOME/.cache $HOME/.cache
rm -rf ~/.vscode-server
ln -s $SCRATCH_HOME/.vscode-server $HOME/.vscode-server

echo "export SCRATCH_HOME=$SCRATCH_HOME" >> ~/.bashrc
echo "export PROJECT_HOME=$PROJECT_HOME" >> ~/.bashrc

source ~/.bashrc
exec /bin/bash