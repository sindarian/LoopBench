#!/bin/bash

# List of methods to test
# if any of these fail, run them separately
methods=("numpy" "csr" "csc" "lil" "hdf5")
#methods=("numpy")

PROJECT_ROOT="/at3462/GILoop"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Loop over each method and call the Python script
for method in "${methods[@]}"
do
    echo "=============================="
    echo "Running method: $method"
    echo "=============================="
    echo ""

    /opt/conda/bin/conda run -p /at3462/conda_envs/GIL --no-capture-output python /at3462/GILoop/profiling/interaction_matrix_creation.py $method

    echo ""
done
