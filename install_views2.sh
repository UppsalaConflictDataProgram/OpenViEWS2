#!/usr/bin/env bash

# Stop on error
set -e
echo "Started installing views."
echo "Initalising conda for this shell"
eval "$(conda shell.bash hook)"

echo "Updating conda"
conda update --all --yes
echo "Removing existing views2 env"
conda remove --name views2 --all --yes
echo "Creating env from env_static.yaml"
# @TODO: Change back to env_static.yaml asap when we have working "builds" for linux
conda env create -f misc/environment.yaml
echo "Activating env"
conda activate views2
echo "Running pip install --editable . to install the views package"
pip install --editable .

echo "Creating storage directory here"
mkdir -p ./storage

# Copy the default config file to default config dir ~/.views2/
if [ ! -f ./config.yaml ];
    then
        echo "No current ./config.yaml found, copying the defaults"
        cp ./misc/defaults.yaml ./config.yaml
    else
        echo "./config.yaml already exists, not changing it"
fi

echo "Great success, you can now do \" conda activate views2 \" in your shell and get started."