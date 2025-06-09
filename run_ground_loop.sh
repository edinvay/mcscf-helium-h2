#!/bin/bash

export OMP_NUM_THREADS=20



# H2
mkdir -p experiments/h2
for N_orbitals in {4..6}
do
    echo "======================="
    echo "Running H2 with N_orbitals = $N_orbitals"
    echo "======================="
    python -u calculate_ground.py "$N_orbitals" "h2" \
        2>&1 | tee "experiments/h2/output_N_orbitals=${N_orbitals}.log"
done
