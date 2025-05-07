#!/bin/bash


N_orbitals="$1"
molecule_name="$2"

# Validate inputs
if [[ -z "$molecule_name" || -z "$N_orbitals" ]]; then
    echo "Usage: $0 <N_orbitals> <molecule_name: helium|h2>"
    exit 1
fi

# Perform the renaming
for ((k=0; k<N_orbitals; k++)); do
    src="experiments/${molecule_name}/guess_orbital_${k}.tree"
    dst="experiments/${molecule_name}/guess_orbital_${k}(hydrogen_type).tree"

    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
        echo "Copied $src → $dst"
    else
        echo "Warning: $src not found"
    fi
done
