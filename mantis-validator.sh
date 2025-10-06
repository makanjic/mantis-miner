#!/bin/bash

WORKDIR=$(readlink -f $(dirname $0))
cd $WORKDIR
source .venv/bin/activate

python3 validator.py --do_save
