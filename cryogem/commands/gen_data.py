import mrcfile, os, json, cv2, time, argparse, logging 
import numpy as np
from tqdm import tqdm
from os import path as osp
from multiprocessing import Pool
from cryogem.utils_gen_data import generate_particles_homo, generate_particles_hetero, image2mask, paste_image

import tempfile
import shutil
import gc


logger = logging.getLogger(__name__)

def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

def warnexists(out):
    if os.path.exists(out):
        logger.warning("Warning: {} already exists. Overwriting.".format(out))
        
def add_args(parser):
    # homo / hetero
    parser.add_argument("--mode", type=str, required=True, choices=["homo", "hetero"], help="use cryodrgn if using hetero")
    parser.add_argument("--debug", action="store_true", default=False, help="if opened, generate micrograph unparallelly.")
    # homo params
    parser.add_argument("--input_map", type=str, help="Input map file path.")
    parser.add_argument("--symmetry", type=str, default='C1', help="Symmetry of volume. [C1, D7...], used to limit generated pose space")
    # hetero params
    parser.add_argument("--drgn_dir", type=str, help="cryodrgn result directory")
    parser.add_argument("--drgn_epoch", type=int, help="checkpoint index, e.g. 49")
    parser.add_argument("--same_rot", action="store_true", default=False, help="if applied, we generate particles with same orientation but differnt heterogenelity.")
    # i/o params
    parser.add_argument("--device", type=str, default="cuda:0", help="pytorch device, e.g. cpu, cuda, cuda:0...")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save folder.")
    parser.add_argument("--n_threads", type=int, default=10, help="number of threads paralleled to generate micrographs.")
    # micrograph params
    parser.add_argument("--n_micrographs", type=int, default=10, help="Number of micrographs to generate.")    
    parser.add_argument("--micrograph_size", type=int, default=1024, help="Micrograph size. [s s]")
    #! (NOT USED) sample particles within uniform distribution, [min_particles_per_micrograph, max_particles_per_micrograph] 
    # parser.add_argument("--min_particles_per_micrograph", type=int, default=90, help="Minimum number of particles per micrograph.")
    # parser.add_argument("--max_particles_per_micrograph", type=int, default=130, help="Maximum number of particles per micrograph.")
    # sample particles within normal distribution, (mu-2sigma, mu+2sigma)
    parser.add_argument("--particles_mu", type=float, default=100, help="Mean number of particles per micrograph.")
    parser.add_argument("--particles_sigma", type=float, default=14, help="Standard deviation of number of particles per micrograph.")
    # particle params
    parser.add_argument("--n_particles", type=int, default=10000, help="Number of particles to generate.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Particle batch size used for generating micrographs.")
    parser.add_argument("--particle_size", type=int, required=True, help="Required particle size. [D D]")
    parser.add_argument("--particle_collapse_ratio", type=float, default=0.5, help="Particle collapse ratio, used to control the overlap between particles, bigger value means more overlap.")
    parser.add_argument("--mask_threshold", type=float, default=0.9, help="Threshold to generate particle mask.")
    return parser

# Then replace the main function with this optimized version
def main(opt):
    mkbasedir(opt.save_dir)
    warnexists(opt.save_dir)

    save_dir = opt.save_dir
    n_micrographs = opt.n_micrographs
    mode = opt.mode
    mask_threshold = opt.mask_threshold
    
    mics_mrc_dir           = osp.join(save_dir, 'mics_mrc')
    mics_png_dir           = osp.join(save_dir, 'mics_png')
    mics_mask_dir          = osp.join(save_dir, 'mics_mask')
    particles_mask_dir     = osp.join(save_dir, 'particles_mask')
    mics_particle_info_dir = osp.join(save_dir, 'mics_particle_info')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mics_mrc_dir, exist_ok=True)
    os.makedirs(mics_png_dir, exist_ok=True)
    os.makedirs(mics_mask_dir, exist_ok=True)
    os.makedirs(particles_mask_dir, exist_ok=True)
    os.makedirs(mics_particle_info_dir, exist_ok=True)
    
    # Fixed batch size
    opt.batch_size = min(1000, int(opt.particles_mu * 2))
    
    # Process micrographs in smaller batches to save memory
    max_mics_per_batch = 1000  # Reduced from 1000 to avoid memory issues
    num_batches = (n_micrographs + max_mics_per_batch - 1) // max_mics_per_batch
    
    # Create temporary directory for memory-mapped files
    temp_dir = tempfile.mkdtemp(prefix="cryogem_")
    logger.info(f"Using temporary directory for memory-mapped files: {temp_dir}")
    
    # Arrays to track file paths for memory mapped arrays
    rotation_files = []
    particle_files = []
    
    # Process data in batches
    for batch_idx in range(num_batches):
        logger.info(f"Processing batch {batch_idx+1}/{num_batches}")
        
        # Calculate start and end indices for this batch
        start_mic = batch_idx * max_mics_per_batch
        end_mic = min((batch_idx + 1) * max_mics_per_batch, n_micrographs)
        batch_mics = end_mic - start_mic
        
        # Calculate number of particles needed for this batch
        batch_particles = int(opt.particles_mu * batch_mics * 1.2)
        
        logger.info(f"Generating {batch_particles} particles for micrographs {start_mic} to {end_mic-1}")
        
        # Generate just this batch's particles
        if mode == 'homo':
            resized_particles, rotations = generate_particles_homo(
                opt.input_map,
                batch_particles,
                opt.particle_size,
                opt.device,
                symmetry=opt.symmetry
            )
        elif mode == 'hetero':
            resized_particles, rotations, z_values = generate_particles_hetero(
                batch_particles,
                opt.particle_size,
                opt.device,
                opt.drgn_dir,
                opt.drgn_epoch,
                opt.same_rot
            )
        
        # Create memory-mapped files for this batch
        rot_file = os.path.join(temp_dir, f"rotations_batch_{batch_idx}.npy")
        part_file = os.path.join(temp_dir, f"particles_batch_{batch_idx}.npy")
        
        # Save the arrays to disk
        np.save(rot_file, rotations)
        np.save(part_file, resized_particles)
        
        # Track the files
        rotation_files.append(rot_file)
        particle_files.append(part_file)
        
        # Now load with memory mapping for processing
        mem_rotations = np.load(rot_file, mmap_mode='r')
        mem_particles = np.load(part_file, mmap_mode='r')
        
        # Get micrograph names for this batch
        sync_mics_name = [f'sync_mic_{i:04d}' for i in range(start_mic, end_mic)]
        
        if not opt.debug:
            pbar = tqdm(total=len(sync_mics_name), unit='image', 
                      desc=f'Generating synthetic micrographs batch {batch_idx+1}/{num_batches}...')
            
            # Process in smaller sub-batches for better memory management
            process_batch_size = opt.batch_size
            batch_epochs = len(sync_mics_name) // process_batch_size
            if len(sync_mics_name) % process_batch_size != 0: 
                batch_epochs += 1
                
            for epoch in range(batch_epochs):
                # Use a smaller number of threads to reduce memory pressure
                n_threads = min(opt.n_threads, 6)
                random_start = np.random.randint(0, mem_particles.shape[0] - process_batch_size + 1)
                epoch_start = epoch * process_batch_size
                epoch_end = min((epoch + 1) * process_batch_size, len(sync_mics_name))
                
                pool = Pool(n_threads)
                for name in sync_mics_name[epoch_start:epoch_end]:
                    pool.apply_async(worker, args=(
                            name, 
                            mem_particles[random_start:random_start + process_batch_size],
                            mem_rotations[random_start:random_start + process_batch_size],
                            opt.particles_mu,
                            opt.particles_sigma,
                            [opt.micrograph_size, opt.micrograph_size],
                            opt.particle_collapse_ratio,
                            random_start + batch_idx * batch_particles,  # offset for particle indexing
                            mics_mrc_dir,          
                            mics_png_dir,          
                            mics_mask_dir,         
                            particles_mask_dir,    
                            mics_particle_info_dir,
                            mask_threshold,
                        ), 
                        callback=lambda arg: pbar.update(1)
                    )
                pool.close()
                pool.join()
                
                # Force garbage collection after each epoch
                gc.collect()
                
            pbar.close()
        else:
            for name in tqdm(sync_mics_name, unit='image', 
                             desc=f'Generating synthetic micrographs batch {batch_idx+1}/{num_batches}...'):
                worker(
                    name, 
                    mem_particles,
                    mem_rotations,
                    opt.particles_mu,
                    opt.particles_sigma,
                    [opt.micrograph_size, opt.micrograph_size],
                    opt.particle_collapse_ratio,
                    batch_idx * batch_particles,  # offset for particle indexing
                    mics_mrc_dir,          
                    mics_png_dir,          
                    mics_mask_dir,         
                    particles_mask_dir,    
                    mics_particle_info_dir,
                    mask_threshold,
                )
        
        # Explicitly cleanup memory-mapped arrays
        del mem_particles
        del mem_rotations
        gc.collect()
        
    logger.info('All batches processed successfully!')
    
    # Combine all rotations and particles from memory-mapped files
    logger.info('Combining particle data...')
    
    # Load each file to get shapes for pre-allocation
    first_rot = np.load(rotation_files[0])
    first_part = np.load(particle_files[0])
    rot_shape = list(first_rot.shape)
    part_shape = list(first_part.shape)
    
    # Calculate total sizes
    total_particles = 0
    for part_file in particle_files:
        part_shape_i = np.load(part_file).shape
        total_particles += part_shape_i[0]
    
    # Create memory-mapped output files instead of in-memory arrays
    rot_shape[0] = total_particles
    part_shape[0] = total_particles
    
    rot_combined_file = os.path.join(temp_dir, "combined_rotations.npy")
    part_combined_file = os.path.join(temp_dir, "combined_particles.npy")
    
    # Create empty files of the right size
    combined_rotations = np.lib.format.open_memmap(
        rot_combined_file, mode='w+', 
        dtype=first_rot.dtype, shape=tuple(rot_shape))
    
    combined_particles = np.lib.format.open_memmap(
        part_combined_file, mode='w+', 
        dtype=first_part.dtype, shape=tuple(part_shape))
    
    # Copy data from memory-mapped files in smaller chunks
    start_idx = 0
    for rot_file, part_file in zip(rotation_files, particle_files):
        logger.info(f"Processing {rot_file} for combination")
        
        # Load with memory mapping
        rot_data = np.load(rot_file, mmap_mode='r')
        part_data = np.load(part_file, mmap_mode='r')
        batch_size = rot_data.shape[0]
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 1000  # Adjust based on your memory constraints
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk_slice = slice(chunk_start, chunk_end)
            
            # Copy the chunk to the combined array
            dest_slice = slice(start_idx + chunk_start, start_idx + chunk_end)
            combined_rotations[dest_slice] = rot_data[chunk_slice]
            combined_particles[dest_slice] = part_data[chunk_slice]
            
            # Force flush to disk
            combined_rotations.flush()
            combined_particles.flush()
        
        start_idx += batch_size
        
        # Clean up
        del rot_data
        del part_data
        gc.collect()
    
    # Ensure data is written to disk
    combined_rotations.flush()
    combined_particles.flush()
    
    # Save metadata
    opt.resized_particles = tuple(combined_particles.shape)
    opt.rotations = tuple(combined_rotations.shape)
    opt.func = None
    opt._parser = None
    with open(osp.join(save_dir, 'opt.json'), 'w') as f:
        json.dump(vars(opt), f, default=lambda x: str(x) if not isinstance(x, (int, float, str, list, dict, bool, type(None))) else x)

    # Save rotations and resized_particles (copy from memory-mapped file)
    logger.info(f"Saving final rotations and particles to {save_dir}")
    np.save(osp.join(save_dir, 'rotations.npy'), combined_rotations)
    np.save(osp.join(save_dir, 'particles.npy'), combined_particles)
    
    # Cleanup temporary directory
    logger.info(f"Cleaning up temporary files in {temp_dir}")
    del combined_rotations
    del combined_particles
    gc.collect()
    shutil.rmtree(temp_dir)

def worker(name, 
           resized_particles,
           rotations,
           particles_mu,
           particles_sigma,
           micrograph_size,
           particle_collapse_ratio,
           particle_index_offset,
           mics_mrc_dir,          
           mics_png_dir,          
           mics_mask_dir,         
           particles_mask_dir,    
           mics_particle_info_dir,
           mask_threshold,
        ):
    """Worker for each process.

    Args:
        name (str): Sync Micrograph save name.
        opt (dict): Configuration dict. It contains:
            resized_particles (np.ndarray)        : Resized particles. [N, H', W']
            minimum_particles_per_micrograph (int): Minimum particles per micrograph.
            maximum_particles_per_micrograph (int): Maximum particles per micrograph.
            unresized_micrograph_size ([int, int]): Unresized micrograph size. [H, W]
            resize_divide_scale (float)           : Resize scale.
            particle_collapse_ratio (float)       : Particle collapse ratio.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    pN, pH, pW = resized_particles.shape
    mH, mW = micrograph_size[0], micrograph_size[1]
    particle_mask_len = int((1 - particle_collapse_ratio) * pH)
    np.random.seed(int(time.time() * 1000) % 2**32)
    # assert maximum_particles_per_micrograph <= pN, 'maximum_particles_per_micrograph should be less than or equal to pN.'
    # sampled_particles_num = np.random.randint(minimum_particles_per_micrograph, maximum_particles_per_micrograph + 1)
    # sampled_particles_idx = np.random.choice(pN, sampled_particles_num, replace=False)
    sampled_particles_num = np.random.normal(particles_mu, particles_sigma)
    sampled_particles_num = max(particles_mu - 2 * particles_sigma, min(particles_mu + 2 * particles_sigma, sampled_particles_num))
    sampled_particles_num = int(min(sampled_particles_num, pN)) # within 2sigma of (mu, sigma) distribution
    sampled_particles_start_idx = np.random.randint(0, pN-sampled_particles_num)
    sampled_particles_idx = np.arange(sampled_particles_start_idx, sampled_particles_start_idx + sampled_particles_num) # mod pN
    assert np.all(sampled_particles_idx < pN), 'sampled_particles_idx should be less than pN.'
    micrograph            = np.zeros((mH, mW), dtype=resized_particles.dtype)
    micrograph_mask       = np.zeros((mH, mW), dtype=bool)
    particle_center_mask  = np.zeros((mH, mW), dtype=bool)
    particle_infomation   = {}
    for i, idx in enumerate(sampled_particles_idx):
        coords = np.where(micrograph_mask == False)
        if len(coords[0]) == 0:
            logger.warning(f'Wish to generate {sampled_particles_num} particles, but only {i} particles can be generated.')
            break
        coords_x, coords_y = coords[0], coords[1]
        k = np.random.randint(len(coords_x))
        x, y = coords_x[k]-pH//2, coords_y[k]-pW//2
        paste_image(micrograph, resized_particles[idx], x, y)
        xi, xa = max(coords_x[k] - particle_mask_len, 0), min(coords_x[k] + particle_mask_len, mH)
        yi, ya = max(coords_y[k] - particle_mask_len, 0), min(coords_y[k] + particle_mask_len, mW)
        micrograph_mask[xi:xa, yi:ya] = True
        particle_center_mask[coords_x[k], coords_y[k]] = True
        xi, xa = max(coords_x[k] - pH//2, 0), min(coords_x[k] + (pH - pH//2), mH)
        yi, ya = max(coords_y[k] - pW//2, 0), min(coords_y[k] + (pW - pW//2), mW)
        if xa - xi != pH or ya - yi != pW:
            continue
        box = [int(xi), int(yi), int(xa), int(ya)]
        particle_infomation['particle_{}'.format(i)] = \
            {
                'particle_idx': int(idx)+particle_index_offset,
                'center_dim0': int(coords_x[k]),
                'center_dim1': int(coords_y[k]),
                'rotation': rotations[idx].tolist(),
                'box': box,
            }
    micrograph_uint8 = ((micrograph - micrograph.min()) / (micrograph.max() - micrograph.min()) * 255).astype(np.uint8)
    particle_mask = (image2mask(micrograph_uint8, threshold=mask_threshold)*255).astype(np.uint8)
    particle_center_mask = particle_center_mask.astype(np.uint8) * 255
    with mrcfile.new(osp.join(mics_mrc_dir, f'{name}.mrc'), overwrite=True) as f:
        f.set_data(micrograph)
    cv2.imwrite(osp.join(particles_mask_dir, f'{name}.png'), particle_mask)
    cv2.imwrite(osp.join(mics_png_dir, f'{name}.png'), micrograph_uint8.astype(np.uint8))
    cv2.imwrite(osp.join(mics_mask_dir, f'{name}.png'), particle_center_mask)
    with open(osp.join(mics_particle_info_dir, f'{name}.json'), 'w') as f:
        json.dump(particle_infomation, f)
    process_info = f'Processing {name} ...'
    return process_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())







