"""
Author: Qi Pang 
Description: dataset
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List, Tuple, Dict
import random


class BlockDataset(Dataset): 
    def __init__(
        self,
        volume_paths: List[str],
        fault_paths: Optional[List[str]] = None,
        scale_pool: Optional[Tuple[int, ...]] = None,
        scale_prob: Optional[Tuple[float, ...]] = None,
        block_size: Tuple[int, int, int] = (128, 128, 128), 
        stride: Optional[Tuple[int, int, int]] = None,  
        normalize: bool = True,
        norm_type: str = "minmax",  # "minmax" or "meanstd"
        use_augmentation: bool = True,
        augmentation_config: Optional[dict] = None,
        random_crop: bool = False,  
    ):
        """
        Args:
            volume_paths: List of .npy file paths, each is (H, W, D)
            scale_pool: Muti-block size (default is (128, 192, 256))
            scale_prob: Prop of each Muti-block size (default is (0.5, 0.3, 0.2))
            block_size: Size of 3D blocks to extract (D, H, W)
            stride: Sliding window stride, None = no overlap
            normalize: Whether to normalize each volume
            norm_type: "minmax" or "meanstd"
            use_augmentation: Whether to use data augmentation
            random_crop: If True, randomly crop; if False, use sliding window
        """
        self.scale_pool = scale_pool
        self.scale_prob = scale_prob
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.normalize = normalize
        self.norm_type = norm_type
        self.random_crop = random_crop
        
        self.has_fault = fault_paths is not None and len(fault_paths) > 0 
        
        # Setup augmentation
        self.use_augmentation = use_augmentation
        if use_augmentation:
            from utils.augmentations import SeismicAugmentation3D
            self.augmentation = SeismicAugmentation3D()
        else:
            self.augmentation = None
        
        # Load and preprocess volumes
        self.volumes = []
        self.fault_volumes = []
        
        self.blocks = []
        
        for vol_idx, path in enumerate(volume_paths):
            vol = self._load_and_preprocess(path)  # (1, D, H, W)
            self.volumes.append(vol)
            if fault_paths is not None:
                fault_vol = self._load_condition(fault_paths[vol_idx], is_label=True)
                self.fault_volumes.append(fault_vol)
            
            # Compute blocks for this volume
            if not random_crop:
                blocks = self._compute_sliding_blocks(vol, vol_idx)
                self.blocks.extend(blocks)
            else:
                # For random crop, we just need volume indices
                self.blocks.append({'volume_idx': vol_idx})
        
    
    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        # Load from .npy (H, W, D)
        vol = np.load(path).astype(np.float32)
        
        # Normalize
        if self.normalize:
            if self.norm_type == "minmax":
                vmin, vmax = vol.min(), vol.max()
                vol = 2 * (vol - vmin) / (vmax - vmin + 1e-8) - 1
            elif self.norm_type == "meanstd":
                vmean, vstd = vol.mean(), vol.std()
                vol = (vol - vmean) / (vstd + 1e-8)
            else:
                raise ValueError(f"Invalid norm_type: {self.norm_type}")
        
        vol = torch.from_numpy(vol).unsqueeze(0)
        return vol
    
    def _load_condition(self, path: str, is_label: bool = False) -> torch.Tensor:
        vol = np.load(path)
        if is_label:
            vol = (vol > 0).astype(np.float32)
        else:
            vol = vol.astype(np.float32)
        if not is_label and self.normalize:
            if self.norm_type == "minmax":
                vmin, vmax = vol.min(), vol.max()
                if vmax > vmin:
                    vol = 2 * (vol - vmin) / (vmax - vmin + 1e-8) - 1
            elif self.norm_type == "meanstd":
                vmean, vstd = vol.mean(), vol.std()
                if vstd > 0:
                    vol = (vol - vmean) / (vstd + 1e-8)
        
        vol = torch.from_numpy(vol).unsqueeze(0)
        return vol
        
    def _compute_sliding_blocks(
        self, 
        volume: torch.Tensor, 
        vol_idx: int
    ) -> List[Dict]:
        _, D, H, W = volume.shape
        block_d, block_h, block_w = self.block_size
        stride_d, stride_h, stride_w = self.stride
        
        blocks = []
        
        for d_start in range(0, D - block_d + 1, stride_d):
            for h_start in range(0, H - block_h + 1, stride_h):
                for w_start in range(0, W - block_w + 1, stride_w):
                    blocks.append({
                        'volume_idx': vol_idx,
                        'd_start': d_start,
                        'h_start': h_start,
                        'w_start': w_start,
                    })
        
        # Handle boundary (last block may extend beyond)
        if D > block_d:
            blocks.append({
                'volume_idx': vol_idx,
                'd_start': D - block_d,
                'h_start': 0,
                'w_start': 0,
            })
        
        return blocks
    
    def _extract_block(
        self, 
        data: torch.Tensor, 
        d_start: int, 
        h_start: int, 
        w_start: int,
        tmp_block_size: Tuple[int, int, int]=None,
    ) -> torch.Tensor:
        if tmp_block_size is None:
            tmp_block_size = self.block_size
        block_d, block_h, block_w = tmp_block_size
        
        block = data[
            :,
            d_start : d_start + block_d,
            h_start : h_start + block_h,
            w_start : w_start + block_w,
        ]
        
        return block

    def _random_crop_block(self, volume: torch.Tensor,
                           fault_volume: Optional[torch.Tensor] = None,) -> torch.Tensor:
        _, D, H, W = volume.shape
        if self.scale_pool is not None:
            random_block_size = random.choices(self.scale_pool, weights=self.scale_prob, k=1)[0]
            block_d = block_h = block_w = random_block_size
        else:
            block_d, block_h, block_w = self.block_size
            
        # Random starting positions
        if D > block_d:
            d_start = random.randint(0, D - block_d)
        else:
            d_start = 0
        
        if H > block_h:
            h_start = random.randint(0, H - block_h)
        else:
            h_start = 0
        
        if W > block_w:
            w_start = random.randint(0, W - block_w)
        else:
            w_start = 0
        
        block = self._extract_block(volume, d_start, h_start, w_start, 
                                    [block_d, block_h, block_w])
        
        fault_block = None
        if fault_volume is not None:
            fault_block = self._extract_block(fault_volume, d_start, h_start, w_start,
                                             [block_d, block_h, block_w])
        # Pad if necessary
        if block.shape[1:] != [block_d, block_h, block_w]:
            pad_d = block_d - block.shape[1]
            pad_h = block_h - block.shape[2]
            pad_w = block_w - block.shape[3]
            
            block = F.pad(
                block,
                (0, pad_w, 0, pad_h, 0, pad_d),
                mode='reflect'
            )
            
            if fault_block is not None:
                fault_block = F.pad(
                    fault_block,
                    (0, pad_w, 0, pad_h, 0, pad_d),
                    mode='constant', 
                    value=0
                )
            
        scale = self.block_size[0] / block_d
        
        crop_info = {
            'd_start': d_start,
            'h_start': h_start,
            'w_start': w_start,
            'scale': scale,
        }  
        
        if scale == 1:
            return block, fault_block, crop_info
        
        block = F.interpolate(
            block.unsqueeze(0), 
            size=(self.block_size[0],
            self.block_size[0],
            self.block_size[0]),
            mode="trilinear",
            align_corners=False).squeeze(0)
        
        if fault_block is not None:
            fault_block = F.interpolate(
                fault_block.unsqueeze(0),
                size=self.block_size,
                mode="nearest").squeeze(0) 
          
        return block, fault_block, crop_info
    
    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        block_info = self.blocks[idx]
        vol_idx = block_info['volume_idx']
        volume = self.volumes[vol_idx]
        
        fault_volume = self.fault_volumes[vol_idx] if self.has_fault else None
        
        # Extract block
        if self.random_crop:
            block, fault_block, crop_info = self._random_crop_block(volume, fault_volume)
        else:
            block = self._extract_block(
                volume,
                block_info['d_start'],
                block_info['h_start'],
                block_info['w_start'],
            )
            if self.has_fault:
               fault_block = self._extract_block(
                   fault_volume,
                   block_info['d_start'],
                   block_info['h_start'],
                   block_info['w_start'],
               )
            else:
               fault_block = None      
                
        # Apply augmentation
        if self.augmentation is not None:
            rng_state = torch.get_rng_state()
            block = self.augmentation(block)
            if self.has_fault:
                torch.set_rng_state(rng_state)
                fault_block = self.augmentation(fault_block, is_label=True)
                
        if self.has_fault:
            fault_mask = torch.tensor(1, dtype=torch.bool)
        else:
            fault_block = torch.zeros_like(block)   
            fault_mask = torch.tensor(0, dtype=torch.bool)
            
        return {
            'target': block,
            'condition': fault_block,
            'condition_mask': fault_mask,
            'volume_idx': vol_idx,
        }