#!/bin/bash

cd "$(dirname $0)/data"

declare -A data_urls=( 
    [dev.ontonotes.withdep.conll]=https://worksheets.codalab.org/rest/bundles/0x6f3556ec9edf4774a2db0ad88f140fac/contents/blob/dev.ontonotes.withdep.conll
    [test.ontonotes.withdep.conll]=https://worksheets.codalab.org/rest/bundles/0x6f3556ec9edf4774a2db0ad88f140fac/contents/blob/test.ontonotes.withdep.conll
    [train.ontonotes.withdep.conll]=https://worksheets.codalab.org/rest/bundles/0x6f3556ec9edf4774a2db0ad88f140fac/contents/blob/train.ontonotes.withdep.conll
    [sst2-dev.tsv]=https://worksheets.codalab.org/rest/bundles/0x6f3556ec9edf4774a2db0ad88f140fac/contents/blob/sst2-dev.tsv
    [sst2-test.tsv]=https://worksheets.codalab.org/rest/bundles/0x6f3556ec9edf4774a2db0ad88f140fac/contents/blob/sst2-test.tsv
    [sst2-train.tsv]=https://worksheets.codalab.org/rest/bundles/0x6f3556ec9edf4774a2db0ad88f140fac/contents/blob/sst2-train.tsv
)

for filename in ${!data_urls[@]}; do
    url=${data_urls[$filename]}
    curl "$url" -o "$filename"
done
