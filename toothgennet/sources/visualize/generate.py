#!/usr/bin/env python
"""
Random Generation for ToothGenNet

Generates random samples from the latent prior.
Output structure matches interpolation for compatibility with viewer.
"""

import sys
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toothgennet.sources.utils.data import get_data_loaders
from toothgennet.sources.utils.model_utils import get_model
from toothgennet.sources.utils.viz_utils import (
    visualize_point_cloud,
    create_interactive_interpolation_visualization
)
from toothgennet.sources.utils.mesh_utils import (
    save_pointcloud_to_ply,
    batch_generate_meshes
)
import pickle

def decode_from_w_space(model, w, num_points=2048, device='cuda'):
    """
    Decode w-space representation to point cloud.
    """
    with torch.no_grad():
        # Reverse through latent CNF to get z
        z = model.latent_cnf(w, None, reverse=True)  # [B, latent_dim]

        # Sample noise and pass through point CNF conditioned on z
        # w is [B, dim], need to handle batch
        batch_size = w.size(0)
        noise = torch.randn(batch_size, num_points, 3, device=device)
        x = model.point_cnf(noise, z, reverse=True)  # [B, num_points, 3]

    return x, z

def generate_random_samples(checkpoint_path, num_samples=8, num_points=15000, 
                          generate_meshes=True, seed=None, output_dir='outputs/generation',
                          regenerate=False):
    """
    Generate random samples from the prior.
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create args_dict for hashing
    args_dict = {
        'checkpoint_path': str(checkpoint_path),
        'num_samples': num_samples,
        'num_points': num_points,
        'generate_meshes': generate_meshes,
        'seed': seed
    }

    # Create MD5 hash
    # Include random salt if seed is None to ensure unique folder for random generation
    if seed is None:
        import time
        args_dict['timestamp'] = time.time()
    
    args_str = json.dumps(args_dict, sort_keys=True)
    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:12]

    # Create output directory structure
    output_folder = Path(output_dir) / args_hash
    
    # Check if output already exists
    if output_folder.exists() and (output_folder / 'args.json').exists():
        print(f"\nOutput folder already exists: {output_folder}")
        if not regenerate:
            print("Skipping regeneration. Using existing results.")
            return str(output_folder)
        else:
            print("Regenerating output...")
            # Don't delete, just overwrite
            # import shutil
            # shutil.rmtree(output_folder) 
            
    output_folder.mkdir(parents=True, exist_ok=True)
    inferences_folder = output_folder / 'inferences'
    inferences_folder.mkdir(exist_ok=True)
    viz_folder = output_folder / 'visualization'
    viz_folder.mkdir(exist_ok=True)

    # Save args
    with open(output_folder / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = get_model(checkpoint_path, device=device, n_sample_points=2048)
    
    # Sample from prior (w-space)
    # Latent dim is typically 128 for PointFlow, check model.zdim if available
    latent_dim = model.zdim if hasattr(model, 'zdim') else 128
    w_codes = torch.randn(num_samples, latent_dim).to(device)
    
    print(f"Generating {num_samples} samples...")
    
    # Decode
    # Do it in batches if num_points is large to avoid OOM? 
    # 8 * 15000 * 3 floats is small enough for GPU.
    generated_points, z_codes = decode_from_w_space(model, w_codes, num_points=num_points, device=device)
    
    # Convert to numpy
    generated_points = generated_points.cpu().numpy() # [B, N, 3]
    w_codes = w_codes.cpu().numpy()
    z_codes = z_codes.cpu().numpy()
    
    # Save aggregated arrays
    np.save(inferences_folder / 'point_clouds.npy', generated_points)
    np.save(inferences_folder / 'latents.npy', w_codes)
    np.save(inferences_folder / 'priors.npy', z_codes)
    
    # Save visualizations
    print(f"Saving visualizations to {viz_folder}...")
    for i, w_code in enumerate(w_codes):
        heatmap_path = viz_folder / f'latent_interpolation_{i:03d}.png' # Reuse naming for viewer compatibility
        
        if w_code.size == 128:
            w_img = w_code.reshape(8, 16)
        else:
            side = int(np.sqrt(w_code.size))
            if side * side == w_code.size:
                w_img = w_code.reshape(side, side)
            else:
                w_img = w_code.reshape(1, -1)

        fig_h = plt.figure(figsize=(2, 1), dpi=100)
        ax_h = plt.Axes(fig_h, [0., 0., 1., 1.])
        ax_h.set_axis_off()
        fig_h.add_axes(ax_h)
        ax_h.imshow(w_img, cmap='coolwarm', aspect='auto')
        plt.savefig(heatmap_path, bbox_inches=None, pad_inches=0)
        plt.close(fig_h)

    # Generate meshes
    mesh_paths = []
    if generate_meshes:
        print(f"Generating meshes...")
        temp_ply_files = []
        for i, points in enumerate(generated_points):
            temp_ply_path = viz_folder / f'temp_{i:03d}.ply'
            save_pointcloud_to_ply(points, temp_ply_path, verbose=False)
            temp_ply_files.append(str(temp_ply_path))
        
        mesh_paths = batch_generate_meshes(
            temp_ply_files, 
            viz_folder, 
            file_prefix='mesh',
            verbose=True
        )
        
        for p in temp_ply_files:
            Path(p).unlink()

    # Save data.pkl for legacy/completeness
    data = {
        'latents': w_codes,
        'points': generated_points,
        'meshes': mesh_paths,
        'z_codes': z_codes
    }
    with open(output_folder / 'data.pkl', 'wb') as f:
        pickle.dump(data, f)
        
    print(f"\nGeneration Complete!")
    return str(output_folder)

if __name__ == '__main__':
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--output_dir', default='outputs/generation')
    args = parser.parse_args()
    
    generate_random_samples(args.checkpoint_path, args.num_samples, output_dir=args.output_dir)
