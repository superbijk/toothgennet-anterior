import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path to allow imports from modules
sys.path.append(str(Path(__file__).resolve().parents[3]))

from toothgennet.sources.utils.data import get_data_loaders
from toothgennet.sources.utils.model_utils import get_model, PointFlowArgs

# -------------------------------------------------------------------------
# Metrics Implementation
# -------------------------------------------------------------------------

# Import metrics from StructuralLosses (installed via pytorch_pcd_metrics)
from StructuralLosses import (
    compute_all_metrics,
    jsd_between_point_cloud_sets
)

# -------------------------------------------------------------------------
# Main Evaluation Logic
# -------------------------------------------------------------------------

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get checkpoint name and paths
    checkpoint_name = Path(args.checkpoint_path).stem
    output_dir = Path(args.output_dir)
    samples_path = output_dir / 'generated_samples.npy'
    evaluate_data_path = output_dir / 'evaluate_data.csv'

    # Centralized metrics file (parent directory)
    centralized_metrics_path = output_dir.parent / 'checkpoints_metric_data.csv'

    # Check if evaluate_data.csv exists and is complete
    if evaluate_data_path.exists():
        print(f"Found existing evaluate_data.csv, loading...")
        df_batch = pd.read_csv(evaluate_data_path)

        # Aggregate metrics from existing data
        batch_metrics_aggregated = {}
        for col in df_batch.columns:
            if col != 'batch_idx':
                batch_metrics_aggregated[col] = df_batch[col].tolist()

        aggregated_metrics = {k: np.mean(v) for k, v in batch_metrics_aggregated.items()}

        print(f"Loaded {len(df_batch)} batches from evaluate_data.csv")
        print("\nAggregated Metrics:")
        for k, v in aggregated_metrics.items():
            print(f"  {k}: {v:.6f}")
    else:
        # Need to run evaluation
        # Load Data
        print(f"Loading dataset from {args.dataset_path}...")
        data_loaders = get_data_loaders(
            args.dataset_path,
            tooth_category=args.tooth_category,
            n_sample_points=args.n_sample_points,
            test_batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Use validation or test loader based on argument
        loader = data_loaders['val_loader'] if args.split == 'val' else data_loaders['test_loader']

        # Collect reference points (Ground Truth)
        print("Collecting reference points...")
        all_ref_points = []
        for i, data in enumerate(loader):
            ref_points = data['sample_points']
            all_ref_points.append(ref_points)

        all_ref_points = torch.cat(all_ref_points, dim=0)
        num_samples = len(all_ref_points)
        print(f"Total reference samples: {num_samples}")

        # Generate or Load Samples
        if samples_path.exists():
            print(f"Loading existing samples from {samples_path}...")
            all_sample_points = torch.tensor(np.load(samples_path))
            print(f"Loaded {len(all_sample_points)} samples")
        else:
            # Load Model
            print(f"Loading model from {args.checkpoint_path}...")
            model = get_model(args.checkpoint_path, device=device, n_sample_points=args.n_sample_points)

            batch_size = args.batch_size
            num_batches = (num_samples + batch_size - 1) // batch_size
            print(f"Generating samples from prior ({num_samples} samples, {num_batches} batches)...")
            all_sample_points = []

            with torch.no_grad():
                for i in range(num_batches):
                    current_batch_size = min(batch_size, num_samples - i * batch_size)

                    # Sample from model
                    if device.type == 'cuda':
                        gpu_id = device.index if device.index is not None else 0
                    else:
                        gpu_id = None

                    _, x = model.sample(current_batch_size, args.n_sample_points, truncate_std=None, gpu=gpu_id)
                    all_sample_points.append(x.cpu())

                    # Progress logging
                    print(f"  Generated batch {i + 1}/{num_batches} ({current_batch_size} samples)")

            all_sample_points = torch.cat(all_sample_points, dim=0)
            print(f"Generation complete: {len(all_sample_points)} samples")

            # Save generated samples
            np.save(samples_path, all_sample_points.numpy())
            print(f"Saved generated samples to {samples_path}")

        # Compute Metrics Batch-by-Batch
        gen_loader = torch.utils.data.DataLoader(all_sample_points, batch_size=args.batch_size, shuffle=False)
        ref_loader = torch.utils.data.DataLoader(all_ref_points, batch_size=args.batch_size, shuffle=False)

        print(f"Computing metrics ({len(gen_loader)} batches)...")

        batch_results = []
        batch_metrics_aggregated = {}

        for batch_idx, (gen_batch, ref_batch) in enumerate(zip(gen_loader, ref_loader)):
            gen_batch = gen_batch.to(device)
            ref_batch = ref_batch.to(device)

            # Compute metrics for this batch
            batch_metrics = compute_all_metrics(gen_batch, ref_batch, args.batch_size, accelerated_cd=True)

            # Compute JSD
            gen_np = gen_batch.detach().cpu().numpy()
            ref_np = ref_batch.detach().cpu().numpy()
            jsd = jsd_between_point_cloud_sets(gen_np, ref_np)

            # Convert to dict with float values
            batch_result = {'batch_idx': batch_idx}
            for k, v in batch_metrics.items():
                val = v.item() if torch.is_tensor(v) else float(v)
                batch_result[k] = val

                # Accumulate for aggregation
                if k not in batch_metrics_aggregated:
                    batch_metrics_aggregated[k] = []
                batch_metrics_aggregated[k].append(val)

            # Add JSD
            batch_result['jsd'] = float(jsd)
            if 'jsd' not in batch_metrics_aggregated:
                batch_metrics_aggregated['jsd'] = []
            batch_metrics_aggregated['jsd'].append(float(jsd))

            batch_results.append(batch_result)

            # Progress logging - show all metrics
            metric_items = list(batch_result.items())[1:]  # Skip batch_idx
            metric_str = " | ".join([f"{k}:{v:.4f}" for k, v in metric_items])
            print(f"  Batch {batch_idx + 1}/{len(gen_loader)}: {metric_str}")

        # Aggregate metrics (mean across batches)
        aggregated_metrics = {k: np.mean(v) for k, v in batch_metrics_aggregated.items()}

        print("\nAggregated Metrics:")
        for k, v in aggregated_metrics.items():
            print(f"  {k}: {v:.6f}")

        # Save per-batch metrics to evaluate_data.csv
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            df_batch.to_csv(evaluate_data_path, index=False)
            print(f"Saved per-batch metrics to {evaluate_data_path}")

    # Update centralized checkpoints_metric_data.csv
    aggregated_with_checkpoint = {'model_checkpoint': checkpoint_name}
    aggregated_with_checkpoint.update(aggregated_metrics)

    # Load existing centralized file or create new
    if centralized_metrics_path.exists():
        df_centralized = pd.read_csv(centralized_metrics_path)
        # Check if checkpoint already exists
        if checkpoint_name in df_centralized['model_checkpoint'].values:
            # Remove old row and append new one
            df_centralized = df_centralized[df_centralized['model_checkpoint'] != checkpoint_name]
            df_centralized = pd.concat([df_centralized, pd.DataFrame([aggregated_with_checkpoint])], ignore_index=True)
            print(f"Updated existing entry for {checkpoint_name} in {centralized_metrics_path.name}")
        else:
            # Append new row
            df_centralized = pd.concat([df_centralized, pd.DataFrame([aggregated_with_checkpoint])], ignore_index=True)
            print(f"Added new entry for {checkpoint_name} to {centralized_metrics_path.name}")
    else:
        # Create new file
        df_centralized = pd.DataFrame([aggregated_with_checkpoint])
        print(f"Created {centralized_metrics_path.name}")

    df_centralized.to_csv(centralized_metrics_path, index=False)
    print(f"Saved to {centralized_metrics_path}")

def evaluate_single(checkpoint_path, args, checkpoint_idx=None, total_checkpoints=None, device=None):
    """
    Evaluate a single checkpoint
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_name = Path(checkpoint_path).stem

    # Show progress if multiple checkpoints
    if checkpoint_idx is not None and total_checkpoints is not None:
        print(f"\n[{checkpoint_idx + 1}/{total_checkpoints}]")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_name}")
        print(f"{'='*60}")

    # Update args with current checkpoint
    args.checkpoint_path = str(checkpoint_path)
    args.output_dir = str(Path(args.base_output_dir) / checkpoint_name)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Call the main evaluate function
    evaluate(args)


def evaluate_batch(checkpoint_paths, args):
    """
    Evaluate multiple checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Check if using fixed priors
    use_fixed_priors = hasattr(args, 'priors_dir') and args.priors_dir is not None

    if use_fixed_priors:
        priors_dir = Path(args.priors_dir)
        prior_w_path = priors_dir / 'sampled_prior.npy'
        prior_y_path = priors_dir / 'sampled_3d_gaussian.npy'

        if prior_w_path.exists() and prior_y_path.exists():
            print(f"Loading fixed priors...")
            prior_w = np.load(prior_w_path)
            prior_y = np.load(prior_y_path)
            print(f"  Loaded {len(prior_w)} priors (w: {prior_w.shape}, y: {prior_y.shape})")
        else:
            print(f"Fixed priors not found, using random sampling")
            use_fixed_priors = False

    print(f"{'='*60}")
    print(f"Batch Evaluation: {len(checkpoint_paths)} checkpoints")
    print(f"{'='*60}")

    # Loop over checkpoints
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        try:
            evaluate_single(
                checkpoint_path=checkpoint_path,
                args=args,
                checkpoint_idx=idx,
                total_checkpoints=len(checkpoint_paths),
                device=device
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, nargs='+', required=True,
                        help='Path(s) to model checkpoint(s) or directory containing checkpoints')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset pickle')
    parser.add_argument('--output_dir', type=str, required=True, help='Base directory to save results')
    parser.add_argument('--priors_dir', type=str, default=None,
                        help='Directory containing fixed priors (sampled_prior.npy, sampled_3d_gaussian.npy)')
    parser.add_argument('--tooth_category', type=str, default='U1')
    parser.add_argument('--n_sample_points', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])

    args = parser.parse_args()

    # Store base output directory
    args.base_output_dir = args.output_dir

    # Process checkpoint_path argument
    checkpoint_paths = []
    for path_arg in args.checkpoint_path:
        path = Path(path_arg)
        if path.is_dir():
            # If directory, find all checkpoints
            found = sorted(path.glob('checkpoint-*.pt'))
            found = [p for p in found if 'latest' not in p.stem]
            checkpoint_paths.extend(found)
        else:
            # Single checkpoint file
            checkpoint_paths.append(path)

    if not checkpoint_paths:
        print("No checkpoints found!")
        exit(1)

    # Create base output directory
    Path(args.base_output_dir).mkdir(parents=True, exist_ok=True)

    # Single or batch evaluation
    if len(checkpoint_paths) == 1:
        print(f"Evaluating single checkpoint\n")
        evaluate_single(checkpoint_paths[0], args, checkpoint_idx=0, total_checkpoints=1)
    else:
        evaluate_batch(checkpoint_paths, args)
