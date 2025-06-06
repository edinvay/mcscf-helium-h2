#!/bin/bash

export OMP_NUM_THREADS=20

# Helium
mkdir -p experiments/helium
for N_orbitals in {1..3}
do
    echo "=======================" | tee "experiments/helium/output_N_orbitals=${N_orbitals}.log"
    echo "Running He+ with N_orbitals = $N_orbitals" | tee -a "experiments/helium/output_N_orbitals=${N_orbitals}.log"
    echo "=======================" | tee -a "experiments/helium/output_N_orbitals=${N_orbitals}.log"
    python -u single_electron.py "$N_orbitals" "helium" \
        2>&1 | tee -a "experiments/helium/output_N_orbitals=${N_orbitals}.log"
    
    echo "===========================" | tee -a "experiments/helium/output_N_orbitals=${N_orbitals}.log"
    echo "Running Helium with N_orbitals = $N_orbitals" | tee -a "experiments/helium/output_N_orbitals=${N_orbitals}.log"
    echo "===========================" | tee -a "experiments/helium/output_N_orbitals=${N_orbitals}.log"
    python -u calculate_ground.py "$N_orbitals" "helium" \
        2>&1 | tee -a "experiments/helium/output_N_orbitals=${N_orbitals}.log"
done

# H2
mkdir -p experiments/h2
for N_orbitals in {1..10}
do
    echo "=======================" | tee "experiments/h2/output_N_orbitals=${N_orbitals}.log"
    echo "Running H2+ with N_orbitals = $N_orbitals" | tee -a "experiments/h2/output_N_orbitals=${N_orbitals}.log"
    echo "=======================" | tee -a "experiments/h2/output_N_orbitals=${N_orbitals}.log"
    python -u single_electron.py "$N_orbitals" "h2" \
        2>&1 | tee -a "experiments/h2/output_N_orbitals=${N_orbitals}.log"

    echo "=======================" | tee -a "experiments/h2/output_N_orbitals=${N_orbitals}.log"
    echo "Running H2 with N_orbitals = $N_orbitals" | tee -a "experiments/h2/output_N_orbitals=${N_orbitals}.log"
    echo "=======================" | tee -a "experiments/h2/output_N_orbitals=${N_orbitals}.log"
    python -u calculate_ground.py "$N_orbitals" "h2" \
        2>&1 | tee -a "experiments/h2/output_N_orbitals=${N_orbitals}.log"
done
