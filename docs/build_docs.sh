#!/usr/bin/env bash

clear

# Stop on non-zero exit
set -e

echo "Initalising conda for this shell"
eval "$(conda shell.bash hook)"
conda activate views2

echo "Generating docs"
# Clear existing generated docs
rm -f source/*
# Auto-generate new docs
# --module-frist makes Package __init__ come before all the submodules
# See https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html#options
sphinx-apidoc --module-first -o source/ ../views
# Make HTML docs
make html
# Make PDF with latex
make latexpdf