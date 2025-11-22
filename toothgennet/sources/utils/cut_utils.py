#!/usr/bin/env python
"""
Cut Generation and Application Utilities for Tooth Restoration

Implements three types of cuts:
- Horizontal: z > a + bx*x + by*y
- Oblique: z > bias + ax*x + ay*y
- Split: x > threshold or x < threshold
"""

import random
import numpy as np
import torch


def generate_random_cut_condition(cut_type='horizontal', seed=None):
    """
    Generate random cut condition formula.

    Args:
        cut_type: 'horizontal', 'oblique', or 'split'
        seed: Random seed for reproducibility

    Returns:
        tuple: (cut_formula_string, parameters_dict)

    Examples:
        >>> formula, params = generate_random_cut_condition('horizontal')
        >>> print(formula)  # "z > 0.192 + 0.151*x + 0.178*y"
        >>> print(params)   # {'a': 0.192, 'bx': 0.151, 'by': 0.178}
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if cut_type == 'horizontal':
        # Horizontal cut: gentle slope (random direction)
        a = round(random.uniform(0.15, 0.22), 3)
        
        # Randomize slopes
        bx = round(random.uniform(0.15, 0.25) * random.choice([-1, 1]), 3)
        by = round(random.uniform(0.15, 0.25) * random.choice([-1, 1]), 3)

        formula = f"z > {a} + {bx}*x + {by}*y"
        params = {'type': 'horizontal', 'a': a, 'bx': bx, 'by': by}

    elif cut_type == 'oblique':
        # Oblique cut: steep x-axis tilt (random side)
        bias = round(random.uniform(0.22, 0.28), 3)
        
        # Randomize side (sign of ax)
        ax_mag = random.uniform(0.5, 0.7)
        ax_sign = random.choice([-1, 1])
        ax = round(ax_mag * ax_sign, 3)
        
        ay = round(random.uniform(0.2, 0.4), 3)

        # Format with proper sign
        formula = f"z > {bias} {ax:+.3f}*x {ay:+.3f}*y"
        params = {'type': 'oblique', 'bias': bias, 'ax': ax, 'ay': ay}

    elif cut_type == 'split':
        # Split cut: vertical plane
        direction = random.choice(['>', '<'])
        threshold = round(random.uniform(0.0, 0.05), 3)

        formula = f"x {direction} {threshold}"
        params = {'type': 'split', 'direction': direction, 'threshold': threshold}

    else:
        raise ValueError(f"Invalid cut_type '{cut_type}'. Choose 'horizontal', 'oblique', or 'split'.")

    return formula, params


def process_invert_cut_condition(cut_condition):
    """
    Invert cut condition (swap > with < and vice versa).

    Used to get the complementary region of a cut.

    Args:
        cut_condition: Cut condition string (e.g., "z > 0.2 + 0.1*x")

    Returns:
        Inverted condition string

    Example:
        >>> invert_cut_condition("z > 0.2 + 0.1*x")
        "z < 0.2 + 0.1*x"
    """
    if cut_condition is None:
        return None

    # Use temp marker to avoid double-swapping
    cut_condition = cut_condition.replace('>', 'TEMP').replace('<', '>').replace('TEMP', '<')
    return cut_condition


def apply_cut_condition(points, cut_condition):
    """
    Apply cut condition to point cloud and return mask.

    Args:
        points: Point cloud array/tensor [N, 3] or [B, N, 3]
        cut_condition: Cut condition string (e.g., "z > 0.2 + 0.1*x")

    Returns:
        Boolean mask array/tensor

    Example:
        >>> points = torch.randn(1000, 3)
        >>> mask = apply_cut_condition(points, "z > 0.1 + 0.15*x")
        >>> cut_points = points[mask]
    """
    if cut_condition is None:
        # No cut - return all True mask
        if isinstance(points, torch.Tensor):
            return torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        else:
            return np.ones(points.shape[0], dtype=bool)

    # Handle both 2D [N, 3] and 3D [B, N, 3] shapes
    if isinstance(points, torch.Tensor):
        if points.dim() == 3:  # [B, N, 3]
            points = points[0]  # Take first batch
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    elif isinstance(points, np.ndarray):
        if points.ndim == 3:  # [B, N, 3]
            points = points[0]  # Take first batch
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    else:
        raise TypeError(f"Unsupported points type: {type(points)}")

    # Evaluate condition
    mask = eval(cut_condition, {}, {'x': x, 'y': y, 'z': z})

    return mask


def process_x_with_cut_condition(x, cut_condition):
    """
    Apply cut condition mask to point cloud and reshape to [1, M, 3].

    This is the main function used in model forward pass.

    Args:
        x: Point cloud tensor/array [N, 3] or [B, N, 3]
        cut_condition: Cut condition string or None

    Returns:
        Filtered point cloud [1, M, 3] where M <= N

    Example:
        >>> x = torch.randn(1, 2048, 3)
        >>> cut_x = process_x_with_cut_condition(x, "z > 0.2")
        >>> print(cut_x.shape)  # torch.Size([1, ~1024, 3])
    """
    if cut_condition is None:
        # Ensure [1, N, 3] shape
        if isinstance(x, torch.Tensor):
            if x.dim() == 2:
                return x.unsqueeze(0)
            return x
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:
                return np.expand_dims(x, axis=0)
            return x

    # Apply mask
    if isinstance(x, torch.Tensor):
        if x.dim() == 2:  # [N, 3]
            mask = apply_cut_condition(x, cut_condition)
            return x[mask].unsqueeze(0)  # [1, M, 3]
        else:  # [B, N, 3]
            x0 = x[0]
            mask = apply_cut_condition(x0, cut_condition)
            return x0[mask].unsqueeze(0)  # [1, M, 3]

    elif isinstance(x, np.ndarray):
        if x.ndim == 2:  # [N, 3]
            mask = apply_cut_condition(x, cut_condition)
            return np.expand_dims(x[mask], axis=0)  # [1, M, 3]
        else:  # [B, N, 3]
            x0 = x[0]
            mask = apply_cut_condition(x0, cut_condition)
            return np.expand_dims(x0[mask], axis=0)  # [1, M, 3]

    else:
        raise TypeError(f"Unsupported type for x: {type(x)}")


def split_by_cut_condition(points, cut_condition):
    """
    Split point cloud into two parts based on cut condition.

    Args:
        points: Point cloud array/tensor [N, 3]
        cut_condition: Cut condition string

    Returns:
        tuple: (kept_points, excluded_points) both [M, 3]

    Example:
        >>> points = np.random.randn(1000, 3)
        >>> kept, excluded = split_by_cut_condition(points, "z > 0.1")
    """
    mask = apply_cut_condition(points, cut_condition)

    if isinstance(points, torch.Tensor):
        kept = points[mask]
        excluded = points[~mask]
    else:
        kept = points[mask]
        excluded = points[~mask]

    return kept, excluded


def get_cut_statistics(points, cut_condition):
    """
    Get statistics about how many points are kept/excluded by cut.

    Args:
        points: Point cloud array/tensor [N, 3]
        cut_condition: Cut condition string

    Returns:
        dict with 'total', 'kept', 'excluded', 'kept_ratio'
    """
    mask = apply_cut_condition(points, cut_condition)

    if isinstance(mask, torch.Tensor):
        kept_count = mask.sum().item()
    else:
        kept_count = mask.sum()

    total_count = len(points)
    excluded_count = total_count - kept_count

    return {
        'total': total_count,
        'kept': kept_count,
        'excluded': excluded_count,
        'kept_ratio': kept_count / total_count if total_count > 0 else 0.0
    }


def format_cut_condition_readable(cut_condition):
    """
    Format cut condition string in a human-readable way.

    Args:
        cut_condition: Cut condition string

    Returns:
        Formatted string

    Example:
        >>> format_cut_condition_readable("z > 0.192 + 0.151*x + 0.178*y")
        "Horizontal cut: z > 0.192 + 0.151x + 0.178y"
    """
    if 'z >' in cut_condition:
        if '-' in cut_condition and 'x' in cut_condition:
            return f"Oblique cut: {cut_condition.replace('*', '')}"
        else:
            return f"Horizontal cut: {cut_condition.replace('*', '')}"
    elif 'x >' in cut_condition or 'x <' in cut_condition:
        return f"Split cut: {cut_condition}"
    else:
        return f"Custom cut: {cut_condition}"
