#!/usr/bin/env python
"""
Tooth Restoration from Simulated Damage

Implements complete restoration pipeline:
1. Generate random cut (horizontal/oblique/split)
2. Apply cut to simulate damage
3. Optimize latent code to restore missing region
4. Generate high-resolution output (15000 points)
5. Create 4-panel visualization
6. Generate meshes (.obj) and point clouds (.ply)
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toothgennet.sources.utils.data import get_data_loaders
from toothgennet.sources.utils.model_utils import get_model
from toothgennet.sources.utils.cut_utils import (
    generate_random_cut_condition,
    process_x_with_cut_condition,
    apply_cut_condition,
    split_by_cut_condition,
    process_invert_cut_condition,
    get_cut_statistics,
    format_cut_condition_readable
)
from toothgennet.sources.utils.viz_utils import (
    create_8panel_restoration_visualization,
    plot_optimization_trajectory
)
from toothgennet.sources.utils.mesh_utils import (
    save_pointcloud_to_ply,
    generate_mesh_from_pointcloud
)


def compute_chamfer_distance(pcd1, pcd2):
    """
    Compute symmetric Chamfer Distance between two point clouds.

    Args:
        pcd1: Point cloud tensor [1, N, 3] or [N, 3]
        pcd2: Point cloud tensor [1, M, 3] or [M, 3]

    Returns:
        Chamfer distance (float)
    """
    if pcd1.dim() == 3:
        pcd1 = pcd1.squeeze(0)  # [N, 3]
    if pcd2.dim() == 3:
        pcd2 = pcd2.squeeze(0)  # [M, 3]

    # Check for empty point clouds to avoid crash
    if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
        return torch.tensor(0.0, device=pcd1.device, requires_grad=True)

    # pcd1 to pcd2
    dist1 = torch.cdist(pcd1, pcd2)  # [N, M]
    min_dist1 = dist1.min(dim=1)[0]  # [N]
    chamfer1 = min_dist1.mean()

    # pcd2 to pcd1
    dist2 = torch.cdist(pcd2, pcd1)  # [M, N]
    min_dist2 = dist2.min(dim=1)[0]  # [M]
    chamfer2 = min_dist2.mean()

    return (chamfer1 + chamfer2) / 2.0


def optimize_restoration(model, target_points, z_init, cut_condition=None, num_iterations=200, learning_rate=0.01, step_size=100, gamma=0.1, num_subsample=2048, highres_num_points=15000, device='cuda', verbose=True):
    """
    Optimize latent code to restore missing tooth region.
    """
    import time
    start_time = time.time()
    
    if isinstance(target_points, np.ndarray):
        target_points = torch.tensor(target_points).float().to(device)
    
    # Preprocess cut condition: Invert to select the visible (target) region
    # If cut_condition selects the MISSING part, inverted selects the TARGET part.
    cut_condition_inverted = process_invert_cut_condition(cut_condition)

    optimized_zs = []
    losses = []
    reconstructed_points = []

    z = z_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if verbose:
        print(f"Starting optimization: {num_iterations} iterations, LR={learning_rate}, Step={step_size}, Gamma={gamma}")

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Sample Gaussian input
        y = model.sample_gaussian(size=(1, num_subsample, 3), truncate_std=None, gpu=device)  # [1, N, 3]
        x = model.point_cnf(y, z, reverse=True).view(*y.size())  # [1, N, 3]
        
        # Mask generated points to match the target region
        x_visible = process_x_with_cut_condition(x, cut_condition_inverted)

        # Subsample target points (each iteration)
        if target_points.shape[0] > num_subsample:
            if target_points.dim() == 3:
                tp = target_points[0]
            else:
                tp = target_points
            
            if tp.shape[0] > num_subsample:
                idx = np.random.choice(tp.shape[0], num_subsample, replace=False)
                subsample_target_points = tp[idx]
            else:
                subsample_target_points = tp
        else:
             if target_points.dim() == 3:
                 subsample_target_points = target_points[0]
             else:
                 subsample_target_points = target_points

        if subsample_target_points.dim() == 2:
            subsample_target_points = subsample_target_points.unsqueeze(0)

        # Compute loss
        loss = compute_chamfer_distance(x_visible, subsample_target_points)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging and saving
        losses.append(loss.item())
        optimized_zs.append(z.detach().cpu().numpy())
        reconstructed_points.append(x.detach().cpu().numpy())

        if verbose: # Print every iteration
             print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")

    # Generate high-res output
    highres_y = model.sample_gaussian(size=(1, highres_num_points, 3), truncate_std=None, gpu=device)
    highres_reconstructed_points = model.point_cnf(highres_y, z, reverse=True).view(*highres_y.size())
    
    end_time = time.time()
    if verbose:
        print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
    
    return {
        'optimized_zs': optimized_zs,
        'reconstructed_points': reconstructed_points,
        'losses': losses,
        'y': y.detach().cpu().numpy(),
        'highres_y': highres_y.detach().cpu().numpy(),
        'highres_reconstructed_points': highres_reconstructed_points.detach().cpu().numpy(),
        'elapsed_time': end_time - start_time,
        'z_optimized': z.detach()
    }


def restore_tooth(checkpoint_path, dataset_path, sample_idx=0, cut_type='horizontal',
                  cut_formula=None, num_iterations=200, learning_rate=0.01, 
                  step_size=100, gamma=0.1, viz_steps=5, output_points=15000,
                  seed=None, output_dir='outputs/restore_damaged_tooth', random_sample=False):
    """
    Main tooth restoration pipeline.
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate cut condition
    if cut_formula:
        print(f"\nUsing provided cut formula: {cut_formula}")
        cut_params = {"type": "custom"}
    else:
        # Generate random cut
        cut_formula, cut_params = generate_random_cut_condition(cut_type=cut_type, seed=seed)

    # Create MD5 hash for reproducibility
    args_dict = {
        'checkpoint_path': str(checkpoint_path),
        'dataset_path': str(dataset_path),
        'sample_idx': sample_idx,
        'cut_type': cut_type,
        'cut_formula': cut_formula,
        'cut_params': cut_params,
        'num_iterations': num_iterations,
        'learning_rate': learning_rate,
        'step_size': step_size,
        'gamma': gamma,
        'viz_steps': viz_steps,
        'output_points': output_points,
        'seed': seed
    }

    args_str = json.dumps(args_dict, sort_keys=True)
    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:12]

    # Create output directory
    output_folder = Path(output_dir) / args_hash
    
    # Check if output already exists and is complete
    if output_folder.exists() and (output_folder / 'metrics.json').exists():
        print(f"\nOutput folder already exists and is complete: {output_folder}")
        print("Skipping optimization and using existing results.")
        return output_folder
        
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_folder / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    print(f"\nOutput folder: {output_folder}")

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    data_loaders = get_data_loaders(dataset_path, n_sample_points=2048, test_batch_size=1)
    dataset = data_loaders['train_loader'].dataset
    print(f"Dataset size: {len(dataset)} samples")

    # Get original sample
    if random_sample:
        sample_idx = np.random.randint(0, len(dataset))
        print(f"Selected random sample index: {sample_idx}")

    original_points = dataset[sample_idx]['sample_points']  # [N, 3]
    if isinstance(original_points, np.ndarray):
        original_points = torch.from_numpy(original_points).float()
    original_points = original_points.to(device)

    print(f"Sample {sample_idx}: {original_points.shape[0]} points")

    print(f"\nUsing {cut_type} cut:")
    try:
        print(f"Cut formula: {format_cut_condition_readable(cut_formula)}")
    except:
        print(f"Cut formula: {cut_formula}")
    
    # Only print params if not custom
    if cut_params.get('type') != 'custom':
        print(f"Parameters: {cut_params}")

    # Apply cut
    # split_by_cut_condition returns:
    #   kept_points: Points matching condition (e.g. z > 0.2, Top)
    #   excluded_points: Points NOT matching (e.g. z <= 0.2, Bottom)
    kept_points, excluded_points = split_by_cut_condition(original_points, cut_formula)
    
    # Optimization Target:
    # We want to restore the 'kept_points' (Missing Part) using 'excluded_points' (Existing Part) as input.
    # So the target for optimization is the Existing Part.
    target_points = excluded_points
    
    stats = get_cut_statistics(original_points, cut_formula)

    print(f"\nCut statistics:")
    print(f"  Total points: {stats['total']}")
    print(f"  Missing points (to restore): {len(kept_points)} (matches condition)")
    print(f"  Input points (reference): {len(excluded_points)} (opposite of condition)")

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = get_model(checkpoint_path, device=device, n_sample_points=2048)
    print(f"Model loaded successfully")

    # Prepare for optimization
    sampled_prior = model.sample_gaussian(size=(1, model.zdim), truncate_std=None, gpu=device)
    z_init = model.latent_cnf(sampled_prior, None, reverse=True).view(*sampled_prior.size())

    # Optimize restoration
    result = optimize_restoration(
        model=model,
        target_points=target_points,
        z_init=z_init,
        cut_condition=cut_formula,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        step_size=step_size,
        gamma=gamma,
        num_subsample=2048,
        highres_num_points=output_points,
        device=device,
        verbose=True
    )

    # Get high-resolution restored points (full shape)
    restored_full_np = result['highres_reconstructed_points'][0]
    
    # Extract the missing part from the full restoration
    # We want points that MATCH the cut condition (the missing top)
    mask_restored = apply_cut_condition(restored_full_np, cut_formula)
    restored_part_np = restored_full_np[mask_restored]
    
    restored_part = torch.from_numpy(restored_part_np).float().to(device)

    # Create combined point cloud (Input + Restored Part)
    # Input = excluded_points (Bottom)
    # Restored = restored_part (Top)
    combined_points = torch.cat([excluded_points, restored_part], dim=0)

    print(f"Restored full shape: {restored_full_np.shape}")
    print(f"Restored part shape: {restored_part.shape}")
    print(f"Combined shape: {combined_points.shape}")

    # Generate high-resolution points for intermediate steps
    print(f"\nGenerating high-resolution points for intermediate optimization steps...")
    step_indices = [int(i) for i in np.linspace(0, num_iterations - 1, min(viz_steps, num_iterations))]
    optimization_steps = []
    optimized_zs = result['optimized_zs']

    for step_idx in step_indices:
        z_step_np = optimized_zs[step_idx]
        z_step = torch.from_numpy(z_step_np).to(device)
        with torch.no_grad():
            noise = torch.randn(1, output_points, 3, device=device)
            recon = model.point_cnf(noise, z_step, reverse=True)
        
        # For viz, we show the Input (excluded) + Generated Missing (recon masked)
        recon_np = recon.squeeze(0).cpu().numpy()
        mask = apply_cut_condition(recon_np, cut_formula)
        recon_part = recon_np[mask]
        
        # Ensure we have at least some points, otherwise visualization fails
        if len(recon_part) == 0:
            print(f"Warning: Step {step_idx} generated 0 points matching cut condition.")
            # Fallback: use a few dummy points or just the full shape if mask fails completely?
            # Better to just use empty array but log it.
            
        optimization_steps.append(recon_part)
        
    print(f"Generated {len(optimization_steps)} intermediate steps")

    # Convert to numpy for visualization and saving
    original_np = original_points.cpu().numpy()
    target_np = excluded_points.cpu().numpy() # The input reference
    missing_np = kept_points.cpu().numpy()    # The part removed
    combined_np = combined_points.cpu().numpy()

    # Save ... (rest of saving logic needs updates to use correct variables)
    # Note: 'cut_points' in older code referred to what is now 'target_np' (input)?
    # No, in old code cut_points was used as target.
    # I will adhere to new variable names.
    
    # Create output directories
    inputs_dir = output_folder / 'inputs'
    inference_dir = output_folder / 'inference'
    inputs_dir.mkdir(exist_ok=True)
    inference_dir.mkdir(exist_ok=True)
    figures_dir = output_folder / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Save numpy arrays
    print(f"\nSaving numpy arrays...")
    np.save(inputs_dir / 'ref_point_cloud.npy', original_np)
    np.save(inputs_dir / 'input_point_cloud.npy', target_np)
    np.save(inputs_dir / 'missing_point_cloud.npy', missing_np)
    np.save(inputs_dir / 'gaussian_points.npy', result['y'])
    np.save(inputs_dir / 'prior.npy', z_init.detach().cpu().numpy())
    
    np.save(inference_dir / 'z_latent_codes.npy', np.array(result['optimized_zs']))
    
    # Save full shape from optimized latents (all iterations)
    np.save(inference_dir / 'restored_points_from_optimized_latents.npy', np.array(result['reconstructed_points']))
    
    # Save visualization steps (high-res, masked) for viewer
    # This corresponds to "restored_points_damaged_parts.npy" requested by user
    np.save(inference_dir / 'restored_points_damaged_parts.npy', np.array(optimization_steps, dtype=object))

    # PLY Files & Meshes
    # Only save restored mesh as requested
    restored_ply_path = inference_dir / 'restored.ply'
    save_pointcloud_to_ply(restored_part_np, restored_ply_path, verbose=False)

    # Generate meshes
    print(f"Generating meshes...")
    mesh_results = {}
    
    # Only generate restored mesh
    obj_path = inference_dir / 'restored_mesh.obj'
    success = generate_mesh_from_pointcloud(
        restored_ply_path, obj_path,
        samplenum=min(4096, len(combined_np)),
        k_neighbors=128,
        hausdorff_threshold=0.02,
        verbose=False
    )
    mesh_results['restored'] = success
    
    # Remove temporary PLY file
    if restored_ply_path.exists():
        restored_ply_path.unlink()

    # Viz
    print(f"\nCreating visualization...")
    fig = create_8panel_restoration_visualization(
        target_np, missing_np,
        optimization_steps,
        cut_type=cut_type,
        title_prefix=f'Tooth Restoration (Sample {sample_idx})',
        save_path=figures_dir / 'restoration_steps.png',
        step_indices=step_indices,
        num_iterations=num_iterations
    )
    plt.close(fig)

    fig = plot_optimization_trajectory(
        result['losses'],
        save_path=figures_dir / 'optimization_loss.png'
    )
    plt.close(fig)

    # Calculate restoration accuracy (CD between restored missing part and ground truth missing part)
    # restored_part_np vs missing_np
    # Ensure tensors are on device and have correct shape [1, N, 3]
    restored_part_tensor = torch.from_numpy(restored_part_np).float().to(device)
    if restored_part_tensor.dim() == 2:
        restored_part_tensor = restored_part_tensor.unsqueeze(0)
        
    missing_part_tensor = torch.from_numpy(missing_np).float().to(device)
    if missing_part_tensor.dim() == 2:
        missing_part_tensor = missing_part_tensor.unsqueeze(0)
    
    restoration_cd = compute_chamfer_distance(restored_part_tensor, missing_part_tensor).item()
    print(f"Restoration CD (Accuracy): {restoration_cd:.6f}")

    final_cd = result['losses'][-1] if result['losses'] else 0.0
    metrics = {
        'sample_idx': sample_idx,
        'cut_type': cut_type,
        'cut_formula': cut_formula,
        'cut_params': cut_params,
        'original_points': len(original_np),
        'input_points': len(target_np),
        'restored_points': len(restored_part_np),
        'combined_points': len(combined_np),
        'initial_cd': float(result['losses'][0]) if result['losses'] else 0.0,
        'final_cd': float(final_cd),
        'restoration_cd': float(restoration_cd),
        'num_iterations': num_iterations,
        'cut_points': len(target_np) # Legacy key for viewer coloring (first N are input)
    }

    with open(output_folder / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nRestoration Complete!")
    return output_folder


def main():
    parser = argparse.ArgumentParser(description='Tooth Restoration from Simulated Damage')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint (.pt)')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset (.pkl)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to restore (default: 0)')
    parser.add_argument('--cut_type', type=str, default='horizontal',
                       choices=['horizontal', 'oblique', 'split'],
                       help='Type of cut to apply (default: horizontal)')
    parser.add_argument('--num_iterations', type=int, default=200,
                       help='Number of optimization iterations (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Initial learning rate (default: 0.01)')
    parser.add_argument('--step_size', type=int, default=100,
                       help='Step size for LR scheduler (default: 100)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for LR scheduler (default: 0.1)')
    parser.add_argument('--output_points', type=int, default=15000,
                       help='Number of points in output (default: 15000)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--random_sample', action='store_true',
                       help='Select a random sample from the dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/restore_damaged_tooth',
                       help='Output directory')

    args = parser.parse_args()

    restore_tooth(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        sample_idx=args.sample_idx,
        cut_type=args.cut_type,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        step_size=args.step_size,
        gamma=args.gamma,
        output_points=args.output_points,
        seed=args.seed,
        output_dir=args.output_dir,
        random_sample=args.random_sample
    )


if __name__ == '__main__':
    main()