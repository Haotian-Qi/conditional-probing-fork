(2>&1 return 0) > /dev/null

if [[ $? -ne 0 ]]; then
    >&2 echo "You must source this script using 'source $0'"
    return 1
fi

if [[ -z $SGE_CLUSTER_NAME ]]; then
    # Bessemer
    module load Anaconda3/2019.07
    module load GCC/10.2.0
else
    # ShARC
    module load apps/python/conda
    module load dev/gcc/10.1
fi
