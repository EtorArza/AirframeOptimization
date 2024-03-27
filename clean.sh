#!/bin/bash

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <results|cache|logs|nn|all>"
    exit 1
fi

# Perform deletion based on the provided argument
case "$1" in
    results)
        # Delete result data files and comparison figures
        rm results/data/* -f
        rm results/figures/* -rf
        ;;
    solver-cache)
        # Delete solver cache files
        rm cache/*_f.npy -f
        rm cache/*_probstatus.txt -f 
        rm cache/*_x.npy -f
        ;;
    logs)
        # Delete log files
        rm *.log -f
        rm results/data/*.log
        ;;
    nn)
        # Delete neural network model files
        rm cache/*.pth
        ;;
    all)
        # Delete all cache, result data, and figures
        rm cache/* -f
        rm results/data/* -f
        rm results/figures/* -f
        ;;
    *)
        echo "Invalid argument. Please provide 'results', 'cache', 'logs', 'nn', or 'all'."
        exit 1
        ;;
esac

echo "Deletion completed for $1."
