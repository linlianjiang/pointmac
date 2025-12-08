#!/bin/bash

## 2025-03-20
module load cmake/3.27.4 
module load ninja/default
module load singularity/3.11.1
python train_shapenet55.py
