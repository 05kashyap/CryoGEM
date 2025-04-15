#!/bin/bash

# This script is used to run the Python script with the specified arguments.after repo has been cloned

# curl -L -o testing/data.zip https://www.kaggle.com/api/v1/datasets/download/aryankashyapnaveen/cryogem-test

# unzip testing/data.zip -d testing/ # Unzip the downloaded file to the testing directory

# rm -rf testing/data.zip

# python -m cryogem gen_data --mode homo --device cuda:0 \
#   --input_map testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc \
#   --save_dir save_images/ddpm_data/train \
#   --n_micrographs 1000 --particle_size 90 --mask_threshold 0.9

# python -m cryogem gen_data --mode homo --device cuda:0 \
#   --input_map testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc \
#   --save_dir save_images/ddpm_data/test \
#   --n_micrographs 500 --particle_size 90 --mask_threshold 0.9

# python -m cryogem esti_ice --apix 5.36 --input_dir save_images/ddpm_data/train/mics_mrc --save_dir save_images/esti_ice/Ribosome\(10028\)/ --output_len 1024         

# Diffusion timesteps reduced from 1000
python -m cryogem ddpm_pipeline --dataset "Ribosome(10028)" \
  --gpu 0 \
  --training_samples 500 \
  --testing_samples 250 \
  --epochs 2 \
  --timesteps 1000 \
  --batch_size 2 \
  --output_dir save_images/ddpm_experiment