import random, os
from cryogem.datasets.base_dataset import BaseDataset
from cryogem.transform import custom_transform
from cryogem.micrograph import Micrograph
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class CryoGEMDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):

        parser.add_argument('--real_dir', type=str, required=is_train, help='directory contains *.mrc')
        parser.add_argument('--sync_dir', type=str, required=True, help='directory contains *.mrc')
        parser.add_argument('--pose_dir', type=str, required=not is_train, help='directory contains particle information (location/pose) for each micrograph in sync_dir')
        parser.add_argument('--mask_dir', type=str, help='directory contains masks')
        parser.add_argument('--weight_map_dir', type=str, required=True, help='directory contains weight maps')
           
        return parser
        
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.max_dataset_size = opt.max_dataset_size
        self.load_sync_micrographs(opt)
        self.max_dataset_size = min(self.max_dataset_size, len(self.micrographs_A))
        self.load_weight_maps(opt)
        if opt.phase == 'train':
            self.load_real_micrographs(opt)
        
    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index_A = index % self.len_A
            index_B = random.randint(0, self.len_B - 1)
            index_map = random.randint(0, self.len_weight_map - 1)
        else:
            index_A = index
            index_map = index % self.len_weight_map

        # Unpack paths and instantiate Micrograph on-the-fly
        micrograph_A_path, particle_info_path, mask_path = self.micrographs_A[index_A]
        micrograph_A = Micrograph(
            micrograph_A_path,
            particle_info_path=particle_info_path,
            mask_path=mask_path,
        )
        
        if self.opt.phase == 'train':
            micrograph_B = self.micrographs_B[index_B]
            micrograph_B = Micrograph(micrograph_B_path)
        mask_A = micrograph_A.get_mask()
        weight_map = self._load_weight_map(index_map)
        
        A, mask_A, weight_map = custom_transform(
            micrograph_A.get_micrograph(), 
            sync_revs_colr=True, 
            normalize=True, 
            autocontrast=False,
            mask=mask_A,
            weight_map=weight_map,
            crop_size=self.opt.crop_size,
            random_crop=self.opt.phase == 'train'
        )
        A = torch.from_numpy(A).float().unsqueeze(0)
        mask_A = torch.from_numpy(mask_A).unsqueeze(0)
        weight_map = torch.from_numpy(weight_map).float().unsqueeze(0)
        A_path = micrograph_A.get_micrograph_path()
        
        if self.opt.phase == 'train':
            B = custom_transform(
                micrograph_B.get_micrograph(), 
                sync_revs_colr=False, 
                normalize=True, 
                autocontrast=False,
                crop_size=self.opt.crop_size,
                random_crop=self.opt.phase == 'train'
            )
            B = torch.from_numpy(B).float().unsqueeze(0)
            B_path = micrograph_B.get_micrograph_path()
            
            return {
                'A': A,
                'B': B,
                'mask_A': mask_A, 
                'A_paths': A_path,
                'B_paths': B_path,
                'weight_map': weight_map,
            }
            
        else:
            B = torch.zeros_like(A)
            B_path = ''
            rotations, boxes, centers = micrograph_A.get_all_particles_info()
            return {
                'A': A,
                'rotations': rotations,
                'boxes': boxes,
                'centers': centers,
                'mask_A': mask_A,
                'weight_map': weight_map,
            }
            

    def __len__(self):
        if self.opt.phase == 'train':
            return self.opt.max_dataset_size
        else:
            return min(self.opt.num_test, self.len_A)
    
    def load_sync_micrographs(self, opt):
        paths_A = os.listdir(opt.sync_dir)
        paths_A.sort()
        if len(paths_A) > self.max_dataset_size:
            paths_A = paths_A[:self.max_dataset_size]
        self.micrographs_A = []
        for path in tqdm(paths_A, desc='Loading real_A'):
            name = path.split('.')[0]
            micrograph_path = os.path.join(opt.sync_dir, path)
            particle_info_path = os.path.join(opt.pose_dir, name + '.json') if opt.pose_dir else None
            mask_path = os.path.join(opt.mask_dir, name + '.png') if opt.mask_dir else None
            # Store tuple of paths instead of Micrograph object
            self.micrographs_A.append((micrograph_path, particle_info_path, mask_path))
        self.len_A = len(self.micrographs_A)

    def load_real_micrographs(self, opt):
        paths_B = os.listdir(opt.real_dir)
        paths_B.sort()
        if len(paths_B) > self.max_dataset_size:
            if opt.phase == 'train':
                paths_B = paths_B[:self.max_dataset_size]
            else:
                paths_B = paths_B[self.max_dataset_size:self.max_dataset_size+opt.num_test]
        self.micrographs_B = []
        # for path in tqdm(paths_B, desc='Loading real_B'):
        #     micrograph_path     = os.path.join(opt.real_dir, path)
        #     micrograph = Micrograph(
        #         micrograph_path,
        #     )
        #     self.micrographs_B.append(micrograph)
        self.micrographs_B = [os.path.join(opt.real_dir, path) for path in paths_B]
        self.len_B = len(self.micrographs_B)
        
    def load_weight_maps(self, opt):
        paths_map = os.listdir(opt.weight_map_dir)
        paths_map.sort()
        
        # Limit the weight maps to match max_dataset_size
        if len(paths_map) > self.max_dataset_size:
            paths_map = paths_map[:self.max_dataset_size]
        
        logger.info(f"Found {len(paths_map)} weight maps in {opt.weight_map_dir}")
        
        # Store paths instead of loading all images
        self.weight_map_paths = [os.path.join(opt.weight_map_dir, path) for path in paths_map]
        self.len_weight_map = len(self.weight_map_paths)
        
        # Create a cache for recently used weight maps
        self.weight_map_cache = {}
        self.max_cache_size = min(100, self.len_weight_map)  # Cache at most 100 weight maps
        
        # Preload first few weight maps to avoid frequent disk access during initial training
        logger.info(f"Preloading {min(10, self.len_weight_map)} weight maps for faster initial access")
        for i in range(min(10, self.len_weight_map)):
            self._load_weight_map(i)
            
        logger.info(f"Weight map loading complete, using lazy loading for {self.len_weight_map} maps")

    def _load_weight_map(self, index):
        """Helper method to load a single weight map and cache it"""
        if index in self.weight_map_cache:
            return self.weight_map_cache[index]
        
        map_path = self.weight_map_paths[index]
        image = Image.open(map_path).convert('L')
        image = np.array(image)
        image = image / image.max()
        
        # Add to cache and manage cache size
        self.weight_map_cache[index] = image
        if len(self.weight_map_cache) > self.max_cache_size:
            # Remove the least recently used item
            self.weight_map_cache.pop(next(iter(self.weight_map_cache)))
            
        return image