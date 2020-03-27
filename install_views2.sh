#!/usr/bin/env bash

# Stop on error
set -e

echo "Initalising conda for this shell"
eval "$(conda shell.bash hook)"

echo "Updating conda"
conda update --all --yes
echo "Removing existing views2 env"
conda remove --name views2 --all --yes
echo "Creating env from environment.yml"
conda env create -f misc/environment.yml
echo "Activating env"
conda activate views2
echo "Running pip install --editable . to install the views package"
pip install --editable .

echo "Creating sourceme.sh"
echo "# Change options to reflect your environment here" > sourceme.sh
cat misc/defaults.sh | grep export >> sourceme.sh
echo "conda activate views2" >> sourceme.sh

echo "Creating storage directory here"
mkdir -p ./storage

echo "Great success, you can now do \" conda activate views2 \" in your shell and get started."