#!/bin/bash

export OMP_NUM_THREADS=20

# Function to rename general_* files to guess_orbital_*
rename_orbitals() {
    local molecule_name="$1"
    local N_orbitals="$2"

    for ((k=0; k<N_orbitals; k++)); do
        src="experiments/${molecule_name}/general_${N_orbitals}_orbital_phi_${k}.tree"
        dst="experiments/${molecule_name}/guess_orbital_${k}.tree"
        if [[ -f "$src" ]]; then
            cp "$src" "$dst"
            echo "Copied $src → $dst"
        else
            echo "Warning: $src not found"
        fi
    done
}

# Helium
mkdir -p experiments/helium
for N_orbitals in {1..2}
do
    rename_orbitals helium "$N_orbitals"
    echo "==========================="
    echo "Running Helium with N_orbitals = $N_orbitals"
    echo "==========================="
    python -u calculate_ground.py "$N_orbitals" "helium" \
        2>&1 | tee "experiments/helium/output_N_orbitals=${N_orbitals}.log"
done

# H2
mkdir -p experiments/h2
for N_orbitals in {1..2}
do
    rename_orbitals h2 "$N_orbitals"
    echo "======================="
    echo "Running H2 with N_orbitals = $N_orbitals"
    echo "======================="
    python -u calculate_ground.py "$N_orbitals" "h2" \
        2>&1 | tee "experiments/h2/output_N_orbitals=${N_orbitals}.log"
done
