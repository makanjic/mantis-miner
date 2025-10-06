#!/usr/bin/env bash

WORKDIR=$(readlink -f $(dirname $0))
cd $WORKDIR
source .venv/bin/activate

python3 block_timestamp.py --network archive --block $1
