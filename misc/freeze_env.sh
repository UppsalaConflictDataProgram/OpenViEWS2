#!/usr/bin/env bash
echo "Initalising conda for this shell"
eval "$(conda shell.bash hook)"
conda activate views2
conda env export --no-builds | grep -v "prefix" > ../env_static.yaml
