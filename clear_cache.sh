#!/bin/bash
rm -rfv ./build/
find . | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\.o$)" | xargs rm -rf
