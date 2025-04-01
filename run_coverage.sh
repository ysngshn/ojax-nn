#!/bin/sh -
python -m coverage run --source ojnn -m unittest discover -s tests/ -v
python -m coverage report
python -m coverage html
