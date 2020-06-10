#!/usr/bin/env bash

# Overwrite env_static.yaml with the latest versions of depencies from your env.
# Make sure to run all the tests before committing an env_static.yaml with
# newer packages so that we are all working on the same versions.

echo "Initalising conda for this shell"
eval "$(conda shell.bash hook)"
conda activate views2
conda env export --no-builds | grep -v "prefix" > ../env_static.yaml
