import os
import argparse
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument("--dataset", type=str, default="Ribosome(10028)", 
                      help="Dataset name to use")
    parser.add_argument("--gpu", type=str, default="0", 
                      help="GPU device to use")
    parser.add_argument("--training_samples", type=int, default=1000, 
                      help="Number of training samples to generate")
    parser.add_argument("--testing_samples", type=int, default=500, 
                      help="Number of testing samples to generate")
    parser.add_argument("--epochs", type=int, default=100, 
                      help="Number of training epochs")
    parser.add_argument("--timesteps", type=int, default=1000, 
                      help="Number of diffusion timesteps")
    parser.add_argument("--batch_size", type=int, default=8, 
                      help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="save_images/ddpm_experiment", 
                      help="Base directory for output")
    return parser

def run_command(cmd, description):
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"Error output: {result.stderr}")
        sys.exit(1)
    else:
        logger.info(f"Command completed successfully.")
        return result.stdout

def get_input_map_for_dataset(dataset_name):
    base_dir = "testing/data/exp_abinitio_volumes"
    emdb_id = dataset_name.split('(')[1][:-1]
    
    # Look for files matching the pattern
    pattern = f"densitymap.{emdb_id}.*mrc"
    matching_files = [f for f in os.listdir(base_dir) if f.startswith(f"densitymap.{emdb_id}.")]
    
    if not matching_files:
        logger.error(f"No matching densitymap file found for dataset {dataset_name}")
        sys.exit(1)
        
    # Return the first matching file
    return os.path.join(base_dir, matching_files[0])

def main(args):
    # Create base directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Generate training data with CryoGEM
    logger.info("Step 1: Generating training data using CryoGEM...")
    training_data_dir = os.path.join(args.output_dir, "training_data")
    os.makedirs(training_data_dir, exist_ok=True)
    input_map = get_input_map_for_dataset(args.dataset)
    cmd_gen_training = [
        "python", "-m", "cryogem", "gen_data",
        "--mode", "homo",
        "--device", f"cuda:{args.gpu}",
        "--input_map", input_map,
        "--save_dir", training_data_dir,
        "--n_micrographs", str(args.training_samples),
        "--particle_size", "90",
        "--mask_threshold", "0.9",
        "--n_particles", str(max(1000, int(args.training_samples * 100 * 1.2))),  # Ensure enough particles
        "--batch_size", "8"  # Set a small batch size
    ]
    run_command(cmd_gen_training, "Generate training data")

    logger.info("Step 2: Generating ice gradient weight maps...")
    ice_dir = f"save_images/esti_ice/{args.dataset}/"
    os.makedirs(ice_dir, exist_ok=True)

    cmd_esti_ice = [
        "python", "-m", "cryogem", "esti_ice",
        "--apix", "5.36",
        "--input_dir", os.path.join(training_data_dir, "mics_mrc"),
        "--save_dir", ice_dir,
        "--output_len", "1024",
        "--device", f"cuda:{args.gpu}"
    ]
    run_command(cmd_esti_ice, "Generate ice gradient weight maps")
    
    # 3. Generate testing data with CryoGEM
    logger.info("Step 3: Generating testing data using CryoGEM...")
    testing_data_dir = os.path.join(args.output_dir, "testing_data")
    os.makedirs(testing_data_dir, exist_ok=True)
    
    cmd_gen_testing = [
        "python", "-m", "cryogem", "gen_data",
        "--mode", "homo",
        "--device", f"cuda:{args.gpu}",
        "--input_map", input_map,
        "--save_dir", testing_data_dir,
        "--n_micrographs", str(args.testing_samples),
        "--particle_size", "90",
        "--mask_threshold", "0.9"
    ]
    run_command(cmd_gen_testing, "Generate testing data")
    
        # 4. Train DDPM model on generated data
    logger.info("Step 4: Training DDPM model on generated data...")
    ddpm_checkpoints_dir = os.path.join(args.output_dir, "ddpm_checkpoints")
    os.makedirs(ddpm_checkpoints_dir, exist_ok=True)
    
    model_name = "ddpm_model"
    cmd_train_ddpm = [
        "python", "-m", "cryogem", "train_ddpm",
        "--name", model_name,
        "--phase", "train",
        "--gpu_ids", args.gpu,
        "--sync_dir", os.path.join(training_data_dir, "mics_mrc"),
        "--mask_dir", os.path.join(training_data_dir, "particles_mask"),
        "--pose_dir", os.path.join(training_data_dir, "mics_particle_info"),
        "--real_dir", os.path.join(testing_data_dir, "mics_mrc"),
        "--weight_map_dir", f"save_images/esti_ice/{args.dataset}/",
        "--max_dataset_size", str(int(args.training_samples)),
        "--timesteps", str(args.timesteps),
        "--beta_schedule", "linear", 
        "--n_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--checkpoints_dir", ddpm_checkpoints_dir,  # This is the base checkpoint directory
        "--save_dir", os.path.join(args.output_dir, "training_results"),
        "--lr_policy", "linear",
        "--save_epoch_freq", "10",  # Save checkpoints every 10 epochs
    ]
    run_command(cmd_train_ddpm, "Train DDPM model")
    
    # 5. Generate samples with trained DDPM model
    logger.info("Step 5: Generating samples with trained DDPM model...")
    generated_samples_dir = os.path.join(args.output_dir, "generated_samples") 
    os.makedirs(generated_samples_dir, exist_ok=True)
    
    # Use the latest checkpoint for evaluation
    latest_checkpoint = os.path.join(ddpm_checkpoints_dir, model_name, "latest_net_Diffusion.pth")
    
    cmd_eval_fid = [
        "python", "-m", "cryogem", "eval_fid",
        "--model_path", latest_checkpoint,
        "--real_images_dir", os.path.join(testing_data_dir, "mics_png"),
        "--generated_images_dir", generated_samples_dir,
        "--num_samples", str(args.testing_samples),
        "--batch_size", "16",
        "--image_size", "256",
        "--device", f"cuda:{args.gpu}",
        "--timesteps", str(args.timesteps),
    ]
    run_command(cmd_eval_fid, "Generate and evaluate samples")
    
    logger.info("Pipeline completed! Results are available in: " + args.output_dir)
    logger.info("FID results are available in: " + os.path.join(generated_samples_dir, "fid_results.txt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full DDPM training and evaluation pipeline")
    main(add_args(parser).parse_args())