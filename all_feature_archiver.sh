#!/usr/bin/env bash

WORKDIR=$(readlink -f $(dirname $0))
cd ${WORKDIR}
source .venv/bin/activate

if [ -z "$PYTHONPATH"]; then
    export PYTHONPATH=$WORKDIR
else
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
fi

export CRYTOPANIC_TOKEN=7bee57d9cad891c4831121b6504d534a60f21fd5

python3 all_feature_archiver.py \
  --mempool-csv data/mempool_archive_1m.csv \
  --options-csv data/options_archive_1m.csv \
  --sentiment-csv data/sentiment_archive_1m.csv \
  --combined-csv data/features_1m.csv \
  --enable-sentiment
