import os
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from cryogem.models.ddpm_model import DDPMModel
from cryogem.utils import mkdirs
from cryogem.options import process_opt, base_add_args

# Import PyTorch FID implementation
try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
except ImportError:
    raise ImportError("Please install pytorch-fid: pip install pytorch-fid")

logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = [f for f in os.listdir(path) if f.endswith('.png')]
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img

def compute_statistics_of_path(path, model, batch_size, device):
    dataset = ImageDataset(
        path,
        transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    pred_arr = []
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
            
        # Move to CPU and convert to numpy
        pred = pred.cpu().numpy()
        
        # Reshape to [batch_size, 2048]
        pred = pred.reshape(pred.shape[0], -1)
        
        pred_arr.append(pred)
        
    pred_arr = np.concatenate(pred_arr, axis=0)
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    
    return mu, sigma

def add_args(parser):
    parser.add_argument("--model_path", type=str, default="checkpoints/ddpm/latest_net_Diffusion.pth", 
                      help="path to trained DDPM model")
    parser.add_argument("--real_images_dir", type=str, default="save_images/gen_data/Ribosome(10028)/testing_dataset/mics_png", 
                      help="directory with real images")
    parser.add_argument("--generated_images_dir", type=str, required=True, 
                      help="directory to save generated images")
    parser.add_argument("--num_samples", type=int, default=1000, 
                      help="number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=16, 
                      help="batch size for generating samples")
    parser.add_argument("--image_size", type=int, default=256, 
                      help="size of generated images")
    parser.add_argument("--device", type=str, default="cuda:0", 
                      help="device to use")
    parser.add_argument("--fid_batch_size", type=int, default=64, 
                      help="batch size for FID calculation")
    parser.add_argument("--isTrain", action="store_true", help="if specified, train the model")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                      help="directory to save checkpoints")
    parser.add_argument("--name", type=str, default="ddpm_model",
                      help="name of the model")
    parser.add_argument("--preprocess", type=str, default="resize_and_crop",
                      help="preprocessing method")
    parser.add_argument("--timesteps", type=int, default=250,
                      help="number of timesteps for DDPM")

    return parser

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create directory for generated samples
    mkdirs(args.generated_images_dir)
    
    # Load model
    model = DDPMModel(args)
    model.load_networks("latest",args.model_path, hardcode=True)
    model.eval()
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    
    sample_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(sample_batches)):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        if batch_size <= 0:
            break
            
        with torch.no_grad():
            samples = model.generate_samples(n_samples=batch_size, size=args.image_size)
            
        for j, sample in enumerate(samples):
            sample_idx = i * args.batch_size + j
            sample_np = ((sample.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            img = Image.fromarray(sample_np)
            img.save(os.path.join(args.generated_images_dir, f"sample_{sample_idx:05d}.png"))
    
    logger.info("Generated samples saved to: " + args.generated_images_dir)
    
    # Calculate FID
    logger.info("Calculating FID...")
    
    # Load Inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    
    # Calculate statistics for real and generated images
    logger.info("Computing statistics for real images...")
    m1, s1 = compute_statistics_of_path(args.real_images_dir, inception_model, args.fid_batch_size, device)
    
    logger.info("Computing statistics for generated images...")
    m2, s2 = compute_statistics_of_path(args.generated_images_dir, inception_model, args.fid_batch_size, device)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    logger.info(f"FID: {fid_value:.2f}")
    
    with open(os.path.join(args.generated_images_dir, "fid_results.txt"), "w") as f:
        f.write(f"FID between real and generated images: {fid_value:.4f}\n")
        f.write(f"Real images directory: {args.real_images_dir}\n")
        f.write(f"Generated images directory: {args.generated_images_dir}\n")
        f.write(f"Number of generated samples: {args.num_samples}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DDPM using FID")
    main(add_args(parser).parse_args())