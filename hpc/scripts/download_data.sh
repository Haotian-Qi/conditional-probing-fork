#!/bin/bash

DATA_DIR="$(dirname $0)/../../data"
mkdir -p $DATA_DIR
cd $DATA_DIR

# Download ConLL files from Google Drive
CODALAB_URL="https://worksheets.codalab.org/rest/bundles/0x6f3556ec9edf4774a2db0ad88f140fac/contents/blob/"
SST_FILENAMES=(
    "sst2-dev.tsv"
    "sst2-test.tsv"
    "sst2-train.tsv"
)

DRIVE_URL='https://drive.google.com/uc?id=${id}&export=download'
declare -A CONLL_FILES=(
    ["dev.ontonotes.withdep.conll"]="1Wsf3uQkRgyTNLWnEmlDfaiCcsYSNodna"
    ["test.ontonotes.withdep.conll"]="1iHUQcoCPFCB6zmQlcVo04Yuo8b6po1qr"
    ["train.ontonotes.withdep.conll"]="1aDEY5-jzqqoByIOxF5VBCvOfVOGh2nhi"
)

# Download SST
for i in ${!SST_FILENAMES[@]}; do
    filename=${SST_FILENAMES[$i]}
    url="${CODALAB_URL}${filename}"
    echo "Downloading ${filename} from ${url}"
    curl "${url}" -o "${filename}"
done

# Download ConLL from Google Drive
for filename in ${!CONLL_FILES[@]}; do
    # Go to download page for file
    file_id=${CONLL_FILES[$filename]}
    url="$(echo $DRIVE_URL | sed "s/\${id}/$file_id/")"
    echo "Downloading ${filename} from ${url}"

    # Determine if virus checking page is being shown
    content_type=$(curl --head "${url}" | grep -Po 'content-type: \K[-a-z/]*')
    case $content_type in
    "text/html")
        action_url=$(curl -s "${url}" | grep -Po 'action="\K[^"]*' | sed "s/\&amp;/\&/g")
        echo "Redirected to ${action_url}"
        curl -L "${action_url}" -o "${filename}"
        ;;
    "application/binary")
        curl -L "${url}" -o "${filename}"
        ;;
    esac
done
