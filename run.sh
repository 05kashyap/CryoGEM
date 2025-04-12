#!/bin/bash

# This script is used to run the Python script with the specified arguments.

python -m cryogem gen_data --mode homo --device cuda:0 \
  --input_map testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc \
  --save_dir save_images/ddpm_data/train \
  --n_micrographs 100 --particle_size 90 --mask_threshold 0.9

python -m cryogem gen_data --mode homo --device cuda:0 \
  --input_map testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc \
  --save_dir save_images/ddpm_data/test \
  --n_micrographs 50 --particle_size 90 --mask_threshold 0.9

python -m cryogem esti_ice --apix 5.36 \                
  --input_dir save_images/ddpm_data/train/mics_mrc \                     
  --save_dir save_images/esti_ice/Ribosome\(10028\)/ \
  --output_len 1024         

python -m cryogem ddpm_pipeline \           
  --dataset "Ribosome(10028)" \
  --gpu 0 \
  --training_samples 100 \
  --testing_samples 50 \
  --epochs 10 \
  --timesteps 1000 \
  --batch_size 1 \
  --output_dir save_images/ddpm_experiment