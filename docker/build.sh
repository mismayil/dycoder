#!/bin/sh

docker build -f ./docker/Dockerfile --platform=linux/amd64 -t registry.rcp.epfl.ch/nlp/ismayilz/coconut --secret id=my_env,src=/Users/mismayil/Desktop/phd/nlplab/runai/runai_env .