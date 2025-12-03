#!/bin/bash

python -m debugpy --listen localhost:5678 --wait-for-client run_test.py args/gsm_coconut_test.yaml