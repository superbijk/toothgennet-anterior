#!/usr/bin/env python
"""
Visualization Utilities for Point Clouds and Meshes

Ported from legacy notebooks with enhancements for production use.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


def visualize_point_cloud(pcd, title='', pert_order=[0, 1, 2],
                         return_image_array=False, bound=0.5,
                         colors=None, show_ticks=False, show_axes=True,
                         camera=None, alpha=1.0, s=5, **kwargs):
    """
    Visualize point cloud(s) using matplotlib 3D scatter plot.

    Ported from legacy notebook implementation with full compatibility.

    Args:
        pcd: Point cloud or list of point clouds
             - Single: np.array or torch.Tensor [N, 3]
             - Multiple: list of [N1, 3], [N2, 3], ...
        title: Plot title string
        pert_order: Axis permutation order [default: [0,1,2] = xyz]
        return_image_array: If True, return RGBA image array instead of displaying
        bound: Plot bounds, sets limits to [-bound, bound] for all axes
        colors: List of colors for multiple point clouds (e.g., ['red', 'blue'])
        show_ticks: Show axis tick labels
        show_axes: Show axes
        camera: Camera angle [elevation, azimuth] (e.g., [30, 45])
        alpha: Point transparency (0.0 to 1.0)
        s: Point size
        **kwargs: Additional arguments passed to ax.scatter()

    Returns:
        If return_image_array=True: np.array RGBA image [H, W, 4]
        Otherwise: (fig, ax) tuple

    Examples:
        >>> # Single point cloud
        >>> points = np.random.randn(1000, 3)
        >>> fig, ax = visualize_point_cloud(points, title='My Cloud')

        >>> # Multiple point clouds with colors
        >>> p1, p2 = np.random.randn(500, 3), np.random.randn(500, 3)
        >>> img = visualize_point_cloud([p1, p2], colors=['red', 'blue'],
        ...                            return_image_array=True)

        >>> # Custom camera angle
        >>> visualize_point_cloud(points, camera=[45, 60])
    """
    # Convert PyTorch tensor to numpy
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.cpu().detach().numpy()[:, pert_order]
    else:
        # Apply permutation to numpy array or list
        if isinstance(pcd, list):
            for i in range(len(pcd)):
                if isinstance(pcd[i], torch.Tensor):
                    pcd[i] = pcd[i].cpu().detach().numpy()
                pcd[i] = pcd[i][:, pert_order]
        else:
            pcd = pcd[:, pert_order]

    # Create figure
    fig = plt.figure(figsize=(4, 4))
    plt.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Plot single or multiple point clouds
    if isinstance(pcd, list):
        for i, p in enumerate(pcd):
            if colors and i < len(colors):
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=s, color=colors[i], alpha=alpha, **kwargs)
            else:
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=s, alpha=alpha, **kwargs)
    else:
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=s, alpha=alpha, **kwargs)

    # Set bounds
    if bound is not None:
        ax.set_xlim3d([-bound, bound])
        ax.set_ylim3d([-bound, bound])
        ax.set_zlim3d([-bound, bound])

    # Axis visibility
    if not show_axes:
        ax.set_axis_off()

    if not show_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Camera angle
    if camera is not None:
        ax.view_init(elev=camera[0], azim=camera[1])

    # Return image array or figure
    if return_image_array:
        fig.canvas.draw()
        res = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        res = res.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return res
    else:
        return fig, ax


def create_8panel_restoration_visualization(cut_points, excluded_points,
                                            optimization_steps,
                                            cut_type='', title_prefix='',
                                            bound=0.3, camera=[30, 300],
                                            crop_offset=40, save_path=None,
                                            step_indices=None, num_iterations=None):
    """
    Create visualization for tooth restoration process.
    Adapts to the number of optimization steps provided.

    Panels:
        1. Reference Point Cloud (kept + excluded, 2 colors)
        2. Input Point Cloud (kept points only)
        3-N. Latent Optimization Steps (kept + reconstructed)

    Args:
        cut_points: Points kept after cut [M, 3]
        excluded_points: Points removed by cut [K, 3]
        optimization_steps: List of reconstructed point clouds [R, 3] at specific iterations
        cut_type: Type of cut ('horizontal', 'oblique', 'split')
        title_prefix: Prefix for overall title
        bound: Plot bounds (default: 0.3)
        camera: Camera angle [elevation, azimuth] (default: [30, 300])
        crop_offset: Pixels to crop from edges (default: 40)
        save_path: If provided, save figure to this path
        step_indices: List of step numbers corresponding to optimization_steps
        num_iterations: Total number of iterations (for title)

    Returns:
        fig: Matplotlib figure
    """
    num_steps = len(optimization_steps)
    num_panels = 2 + num_steps
    
    # Create subplots dynamically based on number of steps
    fig, axes = plt.subplots(1, num_panels, figsize=(3 * num_panels, 6), dpi=300)
    
    # Ensure axes is iterable if num_panels=1 (unlikely here)
    if num_panels == 1:
        axes = [axes]

    cut_type_cap = cut_type.capitalize() if cut_type else 'Unknown'

    # Panel 1: Reference with cut (kept=blue, excluded=green)
    plt.sca(axes[0])
    plt.title(f'Ref. Point Cloud\n({cut_type_cap} Cut)')
    img = visualize_point_cloud([cut_points, excluded_points],
                                bound=bound, alpha=0.5,
                                return_image_array=True, camera=camera,
                                colors=['tab:blue', 'tab:green'])
    # Crop image to remove borders
    img = img[crop_offset:-crop_offset, crop_offset:-crop_offset, :]
    plt.imshow(img)
    plt.axis('off')

    # Panel 2: Input point cloud (kept points only)
    plt.sca(axes[1])
    plt.title(f'Input Point Cloud\n({cut_type_cap} Cut)')
    img = visualize_point_cloud([cut_points],
                                bound=bound, alpha=0.5,
                                return_image_array=True, camera=camera)
    img = img[crop_offset:-crop_offset, crop_offset:-crop_offset, :]
    plt.imshow(img)
    plt.axis('off')

    # Panels 3-N: Optimization steps (kept + reconstructed)
    if step_indices is None:
        step_indices = list(range(1, num_steps + 1))
    
    if num_iterations is None:
        num_iterations = step_indices[-1] if step_indices else 0

    for i, (reconstructed_points, step_idx) in enumerate(zip(optimization_steps, step_indices)):
        plt.sca(axes[i + 2])
        plt.title(f'Latent Optimization\nStep {step_idx}/{num_iterations}')

        # Show kept points + reconstructed points
        img = visualize_point_cloud([cut_points, reconstructed_points],
                                    bound=bound, alpha=0.5,
                                    return_image_array=True, camera=camera)
        img = img[crop_offset:-crop_offset, crop_offset:-crop_offset, :]
        plt.imshow(img)
        plt.axis('off')

    # Overall title
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    return fig


def create_4panel_restoration_visualization(original_points, cut_points, excluded_points,
                                            restored_points, combined_points,
                                            title_prefix='', bound=0.5, save_path=None):
    """
    Create 4-panel visualization for tooth restoration process.

    DEPRECATED: Use create_8panel_restoration_visualization() for legacy notebook format.

    Panels:
        1. Original Point Cloud
        2. Cut Mask Applied (kept + excluded regions colored)
        3. Model Output (Restoration)
        4. Combined Original and Restored

    Args:
        original_points: Original complete tooth [N, 3]
        cut_points: Points kept after cut [M, 3]
        excluded_points: Points removed by cut [K, 3]
        restored_points: Restored/reconstructed points [R, 3]
        combined_points: Cut points + restored points [M+R, 3]
        title_prefix: Prefix for overall title
        bound: Plot bounds
        save_path: If provided, save figure to this path

    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 6))

    # Panel 1: Original
    plt.subplot(1, 4, 1)
    plt.title('Original Point Cloud')
    img1 = visualize_point_cloud(original_points, bound=bound, alpha=0.5,
                                 return_image_array=True)
    plt.imshow(img1)
    plt.axis('off')

    # Panel 2: Cut Applied (show kept + excluded with different colors)
    plt.subplot(1, 4, 2)
    plt.title('Cut Mask Applied\n(Input to Model)')
    img2 = visualize_point_cloud([cut_points, excluded_points],
                                 colors=['blue', 'red'],
                                 bound=bound, alpha=0.5,
                                 return_image_array=True)
    plt.imshow(img2)
    plt.axis('off')

    # Panel 3: Restored
    plt.subplot(1, 4, 3)
    plt.title('Model Output\n(Reconstruction)')
    img3 = visualize_point_cloud(restored_points, bound=bound, alpha=0.5,
                                 return_image_array=True)
    plt.imshow(img3)
    plt.axis('off')

    # Panel 4: Combined
    plt.subplot(1, 4, 4)
    plt.title('Combined\n(Original + Restored)')
    img4 = visualize_point_cloud([cut_points, restored_points],
                                 colors=['blue', 'green'],
                                 bound=bound, alpha=0.5,
                                 return_image_array=True)
    plt.imshow(img4)
    plt.axis('off')

    # Overall title
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 4-panel visualization to: {save_path}")

    return fig


    return fig


def render_mesh_headless(mesh_path, output_size=(400, 400), camera_position=None):
    """
    Render a mesh file to an image using Trimesh and Matplotlib (No OpenGL required).
    This is a robust fallback for headless environments like WSL.
    
    Args:
        mesh_path: Path to .obj or .ply file
        output_size: Image size (W, H)
        camera_position: Camera position list/tuple (ignored in this simple renderer, uses default view)
        
    Returns:
        img: Numpy array [H, W, 3]
    """
    import trimesh
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d
    
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Create figure
    dpi = 100
    fig = plt.figure(figsize=(output_size[0]/dpi, output_size[1]/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create collection
    # Matplotlib's trisurf is slow for large meshes, so we use Poly3DCollection directly for better control
    # or simplify the mesh for visualization if it's too large
    if len(mesh.faces) > 10000:
        # Simplify for visualization speed if needed, though trimesh simplification might need extra deps
        # simpler: just take a subset of faces or use plot_trisurf which handles it reasonably
        pass

    # Normalize vertices to fit in unit box for consistent visualization
    vertices = mesh.vertices
    vertices = vertices - mesh.centroid
    max_range = np.abs(vertices).max()
    vertices = vertices / max_range
    
    # Plot using trisurf (easiest for surfaces)
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   triangles=mesh.faces,
                   cmap='gray', edgecolor='none', alpha=1.0, shade=True,
                   linewidth=0, antialiased=False)
    
    # Setup camera/view
    # Adjust elevation and azimuth to face facial surface
    # Typically facial surface is front, so we might need to rotate
    ax.view_init(elev=10, azim=90) 
    ax.set_axis_off()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    
    # Add a light source (Matplotlib 3D lighting is limited but we can try)
    # The 'shade=True' in plot_trisurf provides basic shading based on normals
    # We can try to improve the material properties
    
    # Render to image
    fig.canvas.draw()
    
    # Convert to numpy array
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    plt.close(fig)
    
    # Return RGB (drop alpha)
    return img[:, :, :3]


def create_3row_interpolation_visualization(meshes_or_images, point_clouds, latent_codes,
                                           num_display=10, title_prefix='',
                                           save_path=None, labels=None):
    """
    Create 3-row visualization for interpolation sequence (legacy notebook format).

    Rows:
        1. Tooth meshes (rendered .obj or images)
        2. Point clouds (colored by z-coordinate)
        3. Latent code heatmaps (reshaped to 8x16 for 128-dim latents)

    Args:
        meshes_or_images: List of mesh images or renderings [num_steps, H, W, 3/4]
        point_clouds: List of point cloud arrays [num_steps, N, 3]
        latent_codes: Latent code array [num_steps, latent_dim] (typically 128)
        num_display: Number of steps to display (evenly spaced, default: 10)
        title_prefix: Prefix for overall title
        save_path: If provided, save figure to this path
        labels: List of labels for the start and end columns (e.g., ['A', 'B'])

    Returns:
        fig: Matplotlib figure

    Notes:
        - Latent codes are reshaped to (8, 16) for 128-dimensional vectors
        - Uses 'coolwarm' colormap for latent heatmaps (legacy format)
        - Figure size: (20, 20) at 300 DPI for high quality
    """
    num_steps = len(point_clouds)

    # Select evenly spaced indices
    if num_steps > num_display:
        indices = np.linspace(0, num_steps - 1, num_display, dtype=int)
    else:
        indices = np.arange(num_steps)
        num_display = num_steps

    fig = plt.figure(figsize=(2 * num_display, 9), dpi=300)

    # Row 1: Meshes
    for i, idx in enumerate(indices):
        plt.subplot(3, num_display, i + 1)
        if meshes_or_images is not None and idx < len(meshes_or_images):
            plt.imshow(meshes_or_images[idx])
        
        title = f'Step {idx}'
        if labels and i == 0:
            title += f' ({labels[0]})'
        elif labels and i == num_display - 1:
            title += f' ({labels[1]})'
            
        plt.title(title)
        plt.axis('off')

    # Row 2: Point clouds
    for i, idx in enumerate(indices):
        plt.subplot(3, num_display, num_display + i + 1)
        pcd = point_clouds[idx]
        img = visualize_point_cloud(pcd, bound=0.5, alpha=0.6,
                                    return_image_array=True, s=3)
        plt.imshow(img)
        plt.axis('off')

    # Row 3: Latent codes (reshaped to 8x16 for 128-dim, use coolwarm colormap)
    for i, idx in enumerate(indices):
        plt.subplot(3, num_display, 2 * num_display + i + 1)
        z = latent_codes[idx]

        # Reshape to 8x16 for 128-dimensional latents (legacy notebook format)
        if len(z) == 128:
            z_reshaped = z.reshape(8, 16)
        else:
            # For other dimensions, reshape to approximate square
            size = int(np.sqrt(len(z)))
            if size * size == len(z):
                z_reshaped = z.reshape(size, size)
            else:
                # Fallback to column vector
                z_reshaped = z.reshape(-1, 1)

        plt.imshow(z_reshaped, cmap='coolwarm', aspect='equal')
        plt.axis('off')

    # Overall title
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=16, y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3-row visualization to: {save_path}")

    return fig


def create_interactive_interpolation_visualization(mesh_paths, point_clouds, latent_codes,
                                                   num_display=10, title_prefix='',
                                                   save_path_html=None, save_path_latents=None,
                                                   labels=None):
    """
    Create interactive visualization for interpolation sequence using Plotly.
    
    Generates:
    1. HTML file with 3 rows: Meshes (top), Point Clouds (middle), Latent Codes (bottom)
    2. Static image for Latent Codes (bottom/separate) - maintained for compatibility
    
    Args:
        mesh_paths: List of paths to .obj files
        point_clouds: List of point cloud arrays [num_steps, N, 3]
        latent_codes: Latent code array [num_steps, latent_dim]
        num_display: Number of steps to display
        title_prefix: Title prefix
        save_path_html: Path to save HTML file
        save_path_latents: Path to save latent codes image
        labels: Start/End labels
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import trimesh
    
    num_steps = len(point_clouds)
    
    # Select indices
    if num_steps > num_display:
        indices = np.linspace(0, num_steps - 1, num_display, dtype=int)
    else:
        indices = np.arange(num_steps)
        num_display = num_steps
        
    # --- 1. Create Interactive Plot (Meshes + Points + Latents) ---
    if save_path_html:
        # Create subplot grid: 3 rows x num_display columns
        specs = [
            [{'type': 'surface'} for _ in range(num_display)],
            [{'type': 'scatter3d'} for _ in range(num_display)],
            [{'type': 'heatmap'} for _ in range(num_display)]
        ]
        
        titles = []
        # Row 1 Titles
        for i in range(num_display):
            t = f'Step {indices[i]}'
            if labels and i == 0: t += f' ({labels[0]})'
            if labels and i == num_display-1: t += f' ({labels[1]})'
            titles.append(t)
        # Row 2 & 3 Titles (empty)
        titles.extend(['' for _ in range(num_display * 2)])
        
        fig = make_subplots(
            rows=3, cols=num_display,
            specs=specs,
            subplot_titles=titles,
            vertical_spacing=0.05,
            horizontal_spacing=0.01,
            row_heights=[0.4, 0.4, 0.2] # Assign less height to heatmap row
        )
        
        # Row 1: Meshes
        print("Generating interactive meshes...")
        for i, idx in enumerate(indices):
            if mesh_paths and idx < len(mesh_paths):
                # Load mesh
                mesh = trimesh.load(mesh_paths[idx], force='mesh')
                # Note: Simplification logic removed to avoid 'Trimesh has no attribute' errors
                
                # Add mesh trace
                fig.add_trace(
                    go.Mesh3d(
                        x=mesh.vertices[:, 0], 
                        y=mesh.vertices[:, 1], 
                        z=mesh.vertices[:, 2],
                        i=mesh.faces[:, 0], 
                        j=mesh.faces[:, 1], 
                        k=mesh.faces[:, 2],
                        color='lightgray',
                        opacity=1.0,
                        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.1, specular=0.8, fresnel=0.2),
                        lightposition=dict(x=0, y=0, z=100)
                    ),
                    row=1, col=i+1
                )
        
        # Row 2: Point Clouds
        print("Generating interactive point clouds...")
        for i, idx in enumerate(indices):
            pcd = point_clouds[idx]
            # Downsample for browser performance if needed
            if len(pcd) > 2048:
                choice = np.random.choice(len(pcd), 2048, replace=False)
                pcd_vis = pcd[choice]
            else:
                pcd_vis = pcd
                
            fig.add_trace(
                go.Scatter3d(
                    x=pcd_vis[:, 0], y=pcd_vis[:, 1], z=pcd_vis[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=pcd_vis[:, 2], # Color by Z
                        colorscale='Viridis',
                        opacity=0.8
                    )
                ),
                row=2, col=i+1
            )

        # Row 3: Latent Heatmaps
        print("Generating interactive latent heatmaps...")
        for i, idx in enumerate(indices):
            code = latent_codes[idx]
            
            # Reshape latent code (assuming 128 dim -> 8x16)
            if code.shape[0] == 128:
                code_img = code.reshape(8, 16)
            else:
                side = int(np.sqrt(code.shape[0]))
                if side * side == code.shape[0]:
                    code_img = code.reshape(side, side)
                else:
                    code_img = code.reshape(1, -1)
            
            fig.add_trace(
                go.Heatmap(
                    z=code_img,
                    colorscale='RdBu',
                    showscale=False, # Hide colorbar to save space
                    zmin=-3, zmax=3  # Fix range for consistency
                ),
                row=3, col=i+1
            )
            
        # Update layout
        fig.update_layout(
            title=title_prefix,
            height=900, # Increased height for 3 rows
            width=300 * num_display,
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Fix camera for 3D scenes (rows 1 & 2)
        # Scenes are named scene, scene2, scene3...
        # There are num_display * 2 3D scenes.
        camera = dict(eye=dict(x=0, y=2.5, z=0)) # Face facial surface (Y-axis)
        for i in range(1, num_display * 2 + 1):
            scene_name = f'scene{i}' if i > 1 else 'scene'
            if scene_name in fig.layout:
                fig.layout[scene_name].update(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=camera,
                    aspectmode='cube'
                )

        # Remove axes for heatmaps (rows 3)
        # Heatmap axes are xy axes (xaxis, yaxis, xaxis2, yaxis2...)
        # We need to find which axes belong to row 3
        # They start after the 3D scenes.
        # 3D scenes don't consume xy axes numbers usually in Plotly, but let's just iterate all xy axes
        for axis in fig.layout:
            if axis.startswith('xaxis') or axis.startswith('yaxis'):
                fig.layout[axis].update(visible=False, showticklabels=False)

        fig.write_html(save_path_html)
        print(f"Saved interactive visualization to {save_path_html}")

    # --- 2. Create Latent Code Plot (Static) ---
    if save_path_latents:
        plt.figure(figsize=(num_display * 2, 3), dpi=300)
        for i, idx in enumerate(indices):
            plt.subplot(1, num_display, i + 1)
            
            # Reshape latent code to 8x16 (assuming 128 dim)
            code = latent_codes[idx]
            if code.shape[0] == 128:
                code_img = code.reshape(8, 16)
            else:
                # Auto-reshape to square-ish
                side = int(np.sqrt(code.shape[0]))
                if side * side == code.shape[0]:
                    code_img = code.reshape(side, side)
                else:
                    code_img = code.reshape(1, -1)
            
            plt.imshow(code_img, cmap='coolwarm', aspect='auto')
            plt.axis('off')
            if i == 0: plt.title('Latent Code')
            
        plt.tight_layout()
        plt.savefig(save_path_latents, bbox_inches='tight')
        plt.close()
        print(f"Saved latent code plot to {save_path_latents}")


def plot_optimization_trajectory(losses, save_path=None):
    """
    Plot the optimization loss trajectory.

    Args:
        losses: List of loss values
        save_path: If provided, save figure to this path

    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Chamfer Distance')
    plt.title('Latent Optimization Trajectory')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved optimization plot to: {save_path}")

    return fig

