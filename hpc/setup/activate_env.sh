(2>&1 return 0) > /dev/null

if [[ $? -ne 0 ]]; then
    >&2 echo "You must source this script using 'source $0'"
    return 1
fi

source activate darwin
