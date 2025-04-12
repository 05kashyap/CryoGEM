import os
import time
import argparse
import logging
import torch
from tqdm import tqdm
import numpy as np

from cryogem.options import process_opt, base_add_args
from cryogem.datasets import create_dataset
from cryogem.models.ddpm_model import DDPMModel
from cryogem.utils import mkdirs, save_as_mrc, save_as_png
from cryogem.visualizer import Visualizer

logger = logging.getLogger(__name__)

def extra_add_args(parser):

    # Required phase parameter
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    
    # Display parameters needed for visualizer
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--update_html_freq', type=int, default=100, help='frequency of saving training results to html')
    parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints')
    
    # Learning rate parameters
    parser.add_argument('--n_epochs_decay', type=int, default=50, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam optimizer')

    parser.add_argument("--timesteps", type=int, default=1000, help="number of diffusion steps")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"], help="noise schedule type")
    parser.add_argument("--sample_interval", type=int, default=1000, help="how often to sample images")
    parser.add_argument("--save_dir", type=str, default="save_images/ddpm", help="directory to save results")
    parser.add_argument("--num_samples", type=int, default=1000, help="number of samples to generate for testing")
    parser.add_argument("--sync_dir", type=str, required=True, help="directory of sync images")
    parser.add_argument("--mask_dir", type=str, required=True, help="directory of particle masks")
    parser.add_argument("--pose_dir", type=str, help="directory of particle location") # Add this line
    parser.add_argument("--real_dir", type=str, help="directory of real images")
    parser.add_argument("--weight_map_dir", type=str, required=True, help="directory of real ice gradients")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to train") 
    parser.add_argument("--epoch_count", type=int, default=0, help="epoch count for resuming training")
    parser.add_argument("--continue_train", action="store_true", help="continue training from a previous checkpoint")
    parser.add_argument("--lr_policy", type=str, default="linear", help="learning rate policy") 
    parser.add_argument("--lr_decay_iters", type=int, default=50, help="multiply by a gamma every lr_decay_iters iterations")
    parser.add_argument("--isTrain", action="store_true", help="if specified, train the model")
    parser.add_argument("--display_id", type=int, default=0, help="window id of the web display")
    parser.add_argument("--display_port", type=int, default=8097, help="port of the web display")
    parser.add_argument("--display_ncols", type=int, default=4, help="number of images per row in the web display")
    # Don't redefine batch_size here as it's already in base_add_args
    
    # Override default values from base_add_args
    parser.set_defaults(batch_size=8)  # Set batch size this way instead
def add_args(parser):
    base_add_args(parser)
    extra_add_args(parser)
    parser.set_defaults(dataset_mode="cryogem", model="ddpm", name="ddpm")
    return parser

def main(args):
    opt = process_opt(args)
    
    # Create directories for saving results
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
        
    logger.info(f"Starting DDPM training with {opt.timesteps} timesteps and {opt.beta_schedule} schedule")
    logger.info(f"Training on data from {opt.sync_dir}")
    
    # Create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    logger.info(f"The number of training images = {dataset_size}")
    
    # Create model
    model = DDPMModel(opt)
    model.setup(opt)
    
    # Create visualizer
    visualizer = Visualizer(opt)
    total_iters = 0
    
    # Create sample directory
    samples_dir = os.path.join(opt.save_dir, "samples")
    mkdirs(samples_dir)
    
    # Training loop
    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
                
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            # Set model input
            model.set_input(data)
            # Update model weights
            model.optimize_parameters()
            
            # Display images
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                # Generate a batch of samples for visualization
                model.forward()
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, save_result)
                
            # Print losses
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                
            # Sample images at regular intervals
            if total_iters % opt.sample_interval == 0:
                with torch.no_grad():
                    samples = model.generate_samples(n_samples=4, size=opt.crop_size)
                    for j, sample in enumerate(samples):
                        sample_np = sample.squeeze().cpu().numpy()
                        save_path = os.path.join(samples_dir, f"iter{total_iters}_sample{j}.png")
                        save_as_png(sample_np, save_path)
                        
                        # Also save as mrc for scientific visualization
                        mrc_path = os.path.join(samples_dir, f"iter{total_iters}_sample{j}.mrc")
                        save_as_mrc(sample_np, mrc_path)
                        
            iter_data_time = time.time()
            
        # Save model at end of epoch
        if epoch % opt.save_epoch_freq == 0:
            logger.info(f"saving the model at the end of epoch {epoch}")
            model.save_networks(epoch)
            
        # Update learning rate
        model.update_learning_rate()
        
        logger.info(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t "+
                   f"Time Taken: {time.time() - epoch_start_time} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())