#!/usr/bin/env python
"""
Reconstruction Visualization for ToothGenNet

Reconstructs point clouds from validation set and saves comparisons.
Output organization: outputs/reconstruction/<md5_hash>/
"""

import argparse
import torch
import numpy as np
import json
import hashlib
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from toothgennet.sources.utils.data import get_data_loaders
from toothgennet.sources.utils.model_utils import get_model


def compute_args_hash(args_dict):
    """Compute MD5 hash of arguments for unique folder naming."""
    sorted_args = json.dumps(args_dict, sort_keys=True)
    return hashlib.md5(sorted_args.encode()).hexdigest()[:12]


def save_point_cloud_comparison(original, reconstructed, filepath, title=""):
    """Save side-by-side comparison of original and reconstructed point clouds."""
    fig = plt.figure(figsize=(20, 10))

    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2],
               c=original[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax1.set_title(f"{title} - Original", fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Reconstructed
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
               c=reconstructed[:, 2], cmap='plasma', s=1, alpha=0.6)
    ax2.set_title(f"{title} - Reconstructed", fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Match axis limits
    all_points = np.concatenate([original, reconstructed], axis=0)
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def chamfer_distance_single(s1, s2):
    """Compute Chamfer distance between two point clouds."""
    s1 = torch.from_numpy(s1).unsqueeze(0).float()
    s2 = torch.from_numpy(s2).unsqueeze(0).float()

    x = s1.unsqueeze(2)  # B x N x 1 x 3
    y = s2.unsqueeze(1)  # B x 1 x M x 3

    dist = torch.sum((x - y) ** 2, dim=-1)  # B x N x M

    min_dist_s1, _ = torch.min(dist, dim=2)  # min over M -> B x N
    min_dist_s2, _ = torch.min(dist, dim=1)  # min over N -> B x M

    cd = torch.mean(min_dist_s1) + torch.mean(min_dist_s2)
    return cd.item()


def reconstruct_samples(args):
    """Main reconstruction function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create args dict for hashing
    args_dict = {
        'checkpoint': Path(args.checkpoint_path).name,
        'dataset': Path(args.dataset_path).name,
        'num_samples': args.num_samples,
        'start_idx': args.start_idx,
        'n_sample_points': args.n_sample_points,
        'seed': args.seed,
    }

    # Compute hash and create output directory
    folder_hash = compute_args_hash(args_dict)
    output_dir = Path(args.output_base_dir) / folder_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Save arguments
    args_file = output_dir / "args.json"
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=2)
    print(f"Saved arguments to: {args_file}")

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    data_loaders = get_data_loaders(
        args.dataset_path,
        n_sample_points=args.n_sample_points,
        test_batch_size=1
    )
    dataset = data_loaders['val_loader'].dataset

    # Validate indices
    max_idx = min(args.start_idx + args.num_samples, len(dataset))
    actual_num_samples = max_idx - args.start_idx

    print(f"Reconstructing {actual_num_samples} samples (indices {args.start_idx} to {max_idx-1})")

    # Load model
    print(f"Loading model from {args.checkpoint_path}...")
    model = get_model(args.checkpoint_path, device=device, n_sample_points=args.n_sample_points)
    model.eval()

    # Reconstruction loop
    all_originals = []
    all_reconstructed = []
    all_chamfer_distances = []

    with torch.no_grad():
        for i in range(args.start_idx, max_idx):
            sample = dataset[i]['sample_points']
            x = torch.from_numpy(sample).unsqueeze(0).float().to(device)

            # Encode and decode
            z = model.encode(x)

            # Decode via point CNF
            noise = torch.randn(1, args.n_sample_points, 3).to(device)
            x_recon = model.point_cnf(noise, z, reverse=True)

            # Convert to numpy
            original = sample
            reconstructed = x_recon.squeeze(0).cpu().numpy()

            # Compute Chamfer distance
            cd = chamfer_distance_single(original, reconstructed)

            # Save
            all_originals.append(original)
            all_reconstructed.append(reconstructed)
            all_chamfer_distances.append(cd)

            # Save comparison image
            filename = f"recon_{i:04d}_CD{cd:.6f}.png"
            save_point_cloud_comparison(
                original, reconstructed,
                output_dir / filename,
                title=f"Sample {i}"
            )

            if (i - args.start_idx + 1) % 10 == 0:
                print(f"  Processed {i - args.start_idx + 1}/{actual_num_samples} samples")

    # Save arrays
    np.save(output_dir / "original_samples.npy", np.stack(all_originals, axis=0))
    np.save(output_dir / "reconstructed_samples.npy", np.stack(all_reconstructed, axis=0))

    # Save metrics
    metrics = {
        'chamfer_distances': all_chamfer_distances,
        'mean_cd': float(np.mean(all_chamfer_distances)),
        'std_cd': float(np.std(all_chamfer_distances)),
        'min_cd': float(np.min(all_chamfer_distances)),
        'max_cd': float(np.max(all_chamfer_distances)),
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create summary plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(args.start_idx, max_idx), all_chamfer_distances, 'o-', alpha=0.7)
    ax.axhline(metrics['mean_cd'], color='r', linestyle='--',
               label=f"Mean: {metrics['mean_cd']:.6f}")
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Chamfer Distance')
    ax.set_title('Reconstruction Quality per Sample')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "chamfer_distance_plot.png", dpi=150)
    plt.close()

    print(f"\n✓ Reconstruction complete!")
    print(f"  Processed {actual_num_samples} samples")
    print(f"  Mean Chamfer Distance: {metrics['mean_cd']:.6f}")
    print(f"  Std Chamfer Distance: {metrics['std_cd']:.6f}")
    print(f"  Output directory: {output_dir}")
    print(f"  Hash: {folder_hash}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Reconstruction visualization for ToothGenNet")

    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset pickle file')
    parser.add_argument('--output_base_dir', type=str,
                       default='outputs/reconstruction',
                       help='Base directory for outputs')

    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to reconstruct')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index in dataset')

    parser.add_argument('--n_sample_points', type=int, default=2048,
                       help='Number of points per sample')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    reconstruct_samples(args)


if __name__ == '__main__':
    main()
