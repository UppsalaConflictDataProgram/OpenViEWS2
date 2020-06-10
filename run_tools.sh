#!/usr/bin/env bash

clear

# Stop on non-zero exit
# We don't want to lint if the tests fail
set -e

echo "Initalising conda for this shell"
eval "$(conda shell.bash hook)"
conda activate views2


echo "Black"
black -l 79 views
black -l 79 projects
black -l 79 tests
black -l 79 runners

echo "mypy views"
mypy views
#echo "mypy projects"
# mypy projects/*
mypy runners
echo "mypy tests"
mypy tests


echo "Running pytest with coverage"
coverage run --source views -m pytest -c misc/pytest.ini tests/
coverage report --show-missing

# Allow non-zero exit for lints
set +e

echo "flake8"
# Ignores are for black conflicts, black wins
flake8 --ignore=E203,W503 views
flake8 --ignore=E203,W503 projects

echo "pylint"
pylint views

echo "Generating docs"
# Clear existing generated docs
rm -f docs/source/*
# Auto-generate new docs
# --module-frist makes Package __init__ come before all the submodules
# See https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html#options
sphinx-apidoc --module-first -o docs/source/ views
# Make HTML docs
make -C docs/ html

git status
