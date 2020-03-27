clear
source activate views2
echo "Black"
black -l 79 views
black -l 79 projects
echo "flake8"
flake8 views
flake8 projects
echo "pylint"
pylint views
echo "mypy views"
mypy views
echo "mypy projects"
mypy projects/*
git status
