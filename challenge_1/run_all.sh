#!/bin/bash
# Run all task scripts sequentially
set -e
cd "$(dirname "$0")"

echo "=== Running Task 1a ==="
python task1a_choose_k.py

echo "=== Running Task 2 ==="
python task2_dft.py

echo "=== Running Task 3 ==="
python task3_validation.py

echo "=== Running Task 4 ==="
python task4_hierarchical.py

echo "=== Running Task 5 ==="
python task5_edit_distance.py

echo "=== All tasks complete ==="
ls -la plots/
