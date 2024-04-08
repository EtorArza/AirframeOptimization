#!/bin/bash

# Check if two arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <results|solver|nn|all> <exclude>"
    exit 1
fi

exclude=$(echo "${2}|.gitkeep" | sed 's/|/\\|/g')

case "$1" in
    results)
        find results/data/ -type f ! -regex '.*\('$exclude'\).*' -delete
        find results/figures/ -type f ! -regex '.*\('$exclude'\).*' -delete
        ;;
    solver)
        find cache/ -type f -name '*_f.npy' ! -regex '.*\('$exclude'\).*' -delete
        find cache/ -type f -name '*_probstatus.txt' ! -regex '.*\('$exclude'\).*' -delete
        find cache/ -type f -name '*_x.npy' ! -regex '.*\('$exclude'\).*' -delete
        find results/data/ -type f -name '*.log' ! -regex '.*\('$exclude'\).*' -delete
        find results/data/ -type f -name '*.csv' ! -regex '.*\('$exclude'\).*' -delete
        ;;
    nn)
        find cache/ -type f -name '*.pth' ! -regex '.*\('$exclude'\).*' -delete
        find . -type f -name '*.log' ! -regex '.*\('$exclude'\).*' -delete
        ;;
    all)
        find cache/ -type f ! -regex '.*\('$exclude'\).*' -delete
        find cache/airframes_animationdata -type f ! -regex '.*\('$exclude'\).*' -delete
        find results/data/ -type f ! -regex '.*\('$exclude'\).*' -delete
        find results/figures/ -type f ! -regex '.*\('$exclude'\).*' -delete
        find . -type f -name '*.log' ! -regex '.*\('$exclude'\).*' -delete
        ;;
    *)
        echo "Invalid argument. Please provide 'results', 'solver', 'nn', 'all'."
        exit 1
        ;;
esac

echo "Deletion completed for $1."
