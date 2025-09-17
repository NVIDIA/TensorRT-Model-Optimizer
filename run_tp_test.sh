#!/bin/bash
# Script to run the test with tensor parallelism

# Set the number of GPUs for tensor parallelism
NUM_GPUS=2

echo "Running Megatron model with Tensor Parallelism (TP=$NUM_GPUS)"
echo "This will use $NUM_GPUS GPUs"

# Check if torchrun is available
if command -v torchrun &> /dev/null; then
    echo "Using torchrun to launch the distributed job..."
    torchrun --nproc_per_node=$NUM_GPUS test.py
else
    echo "torchrun not found, using manual distributed launch..."
    
    # Set environment variables
    export MASTER_ADDR=localhost
    export MASTER_PORT=6001
    export WORLD_SIZE=$NUM_GPUS
    
    # Launch processes
    for ((rank=0; rank<$NUM_GPUS; rank++)); do
        echo "Launching rank $rank..."
        RANK=$rank python test.py &
        pids[$rank]=$!
    done
    
    # Wait for all processes to complete
    for pid in ${pids[*]}; do
        wait $pid
    done
fi

echo "Test completed!"
