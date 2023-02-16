#!/bin/bash

DATA_DIR="$(dirname $0)/data"
ROOT_URL="https://worksheets.codalab.org/rest/bundles/0x6f3556ec9edf4774a2db0ad88f140fac/contents/blob/"
FILENAMES=( 
    "dev.ontonotes.withdep.conll"
    "test.ontonotes.withdep.conll"
    "train.ontonotes.withdep.conll"
    "sst2-dev.tsv"
    "sst2-test.tsv"
    "sst2-train.tsv"
    "dev.ontonotes.withdep.conll.cache.hfacetokensrobertabase.hdf5"
    "test.ontonotes.withdep.conll.cache.hfacetokensrobertabase.hdf5"
    "train.ontonotes.withdep.conll.cache.hfacetokensrobertabase.hdf5"
    "sst2-dev.tsv.cache.hfacetokensrobertabase.hdf5"
    "sst2-test.tsv.cache.hfacetokensrobertabase.hdf5"
    "sst2-train.tsv.cache.hfacetokensrobertabase.hdf5"
)

mkdir -p $DATA_DIR
cd $DATA_DIR

for i in ${!FILENAMES[@]}; do
    filename=${FILENAMES[i]}
    url="${ROOT_URL}${filename}"
    echo "${url}"
    curl "${url}" -o "${filename}"
done
