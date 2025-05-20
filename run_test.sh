#!/bin/bash

export OMP_NUM_THREADS=20

python potential_preparation.py

python initial_orbital_preparation-helium.py

python calculate_ground.py 1 helium