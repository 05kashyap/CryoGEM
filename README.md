# CryoGEM-DDPM 🔬💎: Denoising Diffusion for Cryo-EM Micrograph Generation

**Enhanced CryoGEM with Denoising Diffusion Probabilistic Models (DDPM)**

This repository extends the original [CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy](https://jiakai-zhang.github.io/cryogem/) (NeurIPS 2024) with diffusion models for improved cryo-EM micrograph synthesis.

## Additions

- **DDPM Integration**: Added denoising diffusion probabilistic models for high-quality micrograph generation
- **Comprehensive Evaluation**: Implemented FID-based evaluation pipeline for quantitative assessment
- **End-to-End Pipeline**: Automated workflow from data generation to training and evaluation

## Methodology

### CryoGEM-DDPM Architecture

The approach combines the physics-informed foundations of CryoGEM with diffusion models:

1. **Physics-Based Foundation**: Leverage CryoGEM's accurate CTF simulation and particle physics
2. **Diffusion Training**: Train DDPM on CryoGEM-generated clean micrographs with particle masks
3. **Conditional Generation**: Use particle masks and ice gradients as conditioning information
4. **Quality Enhancement**: Generate more realistic micrographs through iterative denoising

### Training Pipeline

```
Real Data → Ice Estimation → Clean Data Generation → DDPM Training → Evaluation
    ↓              ↓              ↓                    ↓              ↓
CTF Params    Weight Maps    Synthetic Mics      Diffusion Model   FID Scores
```

## Quick Start

### Installation

```bash
# Clone and install CryoGEM-DDPM
git clone <repository-url>
cd CryoGEM
conda create -n cryogem-ddpm python=3.11
conda activate cryogem-ddpm
pip install -e .

# Install additional dependencies for DDPM
pip install pytorch-fid
```

### Dataset Setup

Download the test data:
```bash
curl -L -o testing/data.zip https://www.kaggle.com/api/v1/datasets/download/aryankashyapnaveen/cryogem-test
unzip testing/data.zip
```

### Complete Pipeline

Run the full DDPM training and evaluation pipeline:

```bash
# Automated pipeline for Ribosome dataset
python -m cryogem ddpm_pipeline \
    --dataset "Ribosome(10028)" \
    --gpu "0" \
    --training_samples 1000 \
    --testing_samples 500 \
    --epochs 100 \
    --timesteps 1000 \
    --batch_size 8 \
    --output_dir "save_images/ddpm_experiment"
```

## Experimental Results

### FID Score

| Dataset | Model | Timesteps | Batch Size | Epochs | FID Score | Training Time |
|---------|--------|-----------|------------|--------|-----------|---------------|
| Ribosome(10028) | DDPM (Cosine) | 1000 | 16 | 100 | **109.067** | 30h |

*All experiments conducted on an NVIDIA RTX 4070 with 12GB VRAM.*

Timestep embedding dimension: 1024
Transformer depth: 6 layers
Number of attention heads: 16
Dimension per attention head: 128
MLP expansion ratio: 8
Dropout rate: 0.1
Learning rate: 0.0002 

## Usage

### 1. Generate Training Data with CryoGEM

```bash
# Generate clean micrographs for DDPM training
python -m cryogem gen_data \
    --mode homo \
    --device cuda:0 \
    --input_map testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc \
    --save_dir save_images/ddpm_training_data/ \
    --n_micrographs 1000 \
    --particle_size 90 \
    --mask_threshold 0.9
```

### 2. Estimate Ice Gradients

```bash
# Extract ice gradients from real data
python -m cryogem esti_ice \
    --apix 5.36 \
    --device cuda:0 \
    --input_dir testing/data/Ribosome\(10028\)/real_data/ \
    --save_dir save_images/ice_gradients/
```

### 3. Train DDPM Model

```bash
# Train diffusion model on synthetic data
python -m cryogem train_ddpm \
    --name ribosome_ddpm \
    --gpu_ids 0 \
    --sync_dir save_images/ddmp_training_data/mics_mrc \
    --mask_dir save_images/ddmp_training_data/particles_mask \
    --weight_map_dir save_images/ice_gradients/ \
    --timesteps 1000 \
    --beta_schedule cosine \
    --n_epochs 100 \
    --batch_size 8
```

### 4. Evaluate with FID

```bash
# Generate samples and compute FID scores
python -m cryogem eval_fid \
    --model_path checkpoints/ribosome_ddpm/latest_net_Diffusion.pth \
    --real_images_dir testing/data/Ribosome\(10028\)/real_data/ \
    --generated_images_dir save_images/generated_samples/ \
    --num_samples 1000 \
    --batch_size 16
```

### Training Objectives

The DDPM is trained to denoise micrographs at various timesteps. The loss computed is a combination of L1 and L2 loss.

## Acknowledgments

This work builds upon the excellent foundation provided by:

**Original CryoGEM Team**:
- [Jiakai Zhang](https://jiakai-zhang.github.io)
- [Qihe Chen](https://github.com/Dylan8527) 
- [Yan Zeng](https://zerone182.github.io)
- Wenyuan Gao, Xuming He, Zhijie Liu, Jingyi Yu

**Original Paper**: 
```bibtex
@article{zhang2024cryogem,
  title={CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy},
  author={Zhang, Jiakai and Chen, Qihe and Zeng, Yan and Gao, Wenyuan and He, Xuming and Liu, Zhijie and Yu, Jingyi},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2024}
}
```
