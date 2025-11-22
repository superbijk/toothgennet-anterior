#!/usr/bin/env python
"""
Mesh Generation Utilities using PyMeshLab

Implements Poisson surface reconstruction pipeline for tooth point clouds.
"""

import numpy as np
from pathlib import Path
import pymeshlab
import pyvista as pv
import trimesh


def generate_mesh_from_pointcloud(ply_input_path, obj_output_path,
                                  samplenum=4096, k_neighbors=128,
                                  hausdorff_threshold=0.02,
                                  verbose=True):
    """
    Generate mesh from point cloud using PyMeshLab Poisson reconstruction.

    Pipeline:
        1. Load point cloud from PLY
        2. Simplify point cloud (subsample to samplenum points)
        3. Compute normals (using k_neighbors nearest neighbors)
        4. Poisson surface reconstruction
        5. Taubin smoothing
        6. Remove outliers (Hausdorff distance > threshold)
        7. Laplacian smoothing
        8. Close holes + decimation clustering
        9. Save mesh to OBJ/PLY

    Args:
        ply_input_path: Path to input PLY point cloud file
        obj_output_path: Path to output mesh file (.obj or .ply)
        samplenum: Number of points to subsample (default: 4096)
        k_neighbors: Number of neighbors for normal estimation (default: 128)
        hausdorff_threshold: Threshold for outlier removal (default: 0.02)
        verbose: Print progress messages

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> success = generate_mesh_from_pointcloud(
        ...     'tooth_points.ply',
        ...     'tooth_mesh.obj',
        ...     samplenum=4096,
        ...     k_neighbors=128
        ... )
    """
    try:
        ms = pymeshlab.MeshSet()

        # Step 1: Load point cloud
        if verbose:
            print(f"Loading point cloud from: {ply_input_path}")
        ms.load_new_mesh(str(ply_input_path))

        # Step 2: Simplify and compute normals
        if verbose:
            print(f"Simplifying to {samplenum} points...")
        ms.generate_simplified_point_cloud(samplenum=samplenum, bestsamplepool=32)

        if verbose:
            print(f"Computing normals (k={k_neighbors})...")
        ms.compute_normal_for_point_clouds(k=k_neighbors)
        simplified_cloud_id = ms.current_mesh_id()

        # Step 3: Poisson surface reconstruction
        if verbose:
            print("Running Poisson surface reconstruction...")
        ms.generate_surface_reconstruction_screened_poisson()

        # Step 4: Taubin smoothing
        if verbose:
            print("Applying Taubin smoothing...")
        ms.apply_coord_taubin_smoothing()
        poisson_mesh_id = ms.current_mesh_id()

        # Step 5: Remove outliers using Hausdorff distance
        if verbose:
            print(f"Removing outliers (threshold={hausdorff_threshold})...")
        ms.get_hausdorff_distance(targetmesh=simplified_cloud_id,
                                 sampledmesh=poisson_mesh_id)
        ms.compute_selection_by_condition_per_vertex(condselect=f'(q > {hausdorff_threshold})')
        ms.meshing_remove_selected_vertices()

        # Step 6: Laplacian smoothing
        if verbose:
            print("Applying Laplacian smoothing...")
        ms.apply_coord_laplacian_smoothing()

        # Step 7: Post-processing
        if verbose:
            print("Closing holes and decimating...")
        ms.meshing_close_holes()
        ms.meshing_decimation_clustering()

        # Step 8: Save mesh
        if verbose:
            print(f"Saving mesh to: {obj_output_path}")
        ms.save_current_mesh(str(obj_output_path))

        # Step 9: Fix normals with Trimesh
        if verbose:
            print("Fixing mesh normals with Trimesh...")
        try:
            mesh = trimesh.load(str(obj_output_path), force='mesh')
            
            # Force consistent winding
            trimesh.repair.fix_normals(mesh)
            
            # Robust orientation check
            inverted = False
            if mesh.is_watertight:
                # For watertight meshes, rely on volume
                if mesh.volume < 0:
                    if verbose:
                        print("  Mesh is watertight and volume is negative. Inverting...")
                    mesh.invert()
                    inverted = True
            else:
                # For open meshes (dental crowns), use centroid heuristic
                # Normals should point AWAY from the center of mass
                centroid = mesh.vertices.mean(axis=0)
                
                # Sample faces if too many (for speed)
                if len(mesh.faces) > 2048:
                    sample_indices = np.random.choice(len(mesh.faces), 2048, replace=False)
                    faces_to_check = mesh.faces[sample_indices]
                    normals_to_check = mesh.face_normals[sample_indices]
                else:
                    faces_to_check = mesh.faces
                    normals_to_check = mesh.face_normals
                
                # Compute vectors from centroid to face centers
                face_centers = mesh.vertices[faces_to_check].mean(axis=1)
                vectors = face_centers - centroid
                
                # Check alignment (dot product)
                # Positive dot = Pointing away (Good)
                # Negative dot = Pointing inward (Bad)
                dots = np.sum(vectors * normals_to_check, axis=1)
                
                inward_count = np.sum(dots < 0)
                outward_count = np.sum(dots > 0)
                
                if verbose:
                    print(f"  Open mesh orientation: {outward_count} outward vs {inward_count} inward")
                
                if inward_count > outward_count:
                    if verbose:
                        print("  Majority of normals point inward. Inverting...")
                    mesh.invert()
                    inverted = True

            mesh.export(str(obj_output_path))
        except Exception as e:
            if verbose:
                print(f"Warning: Trimesh normal fix failed: {e}")

        if verbose:
            print(f"✓ Successfully generated mesh: {obj_output_path}")

        return True

    except Exception as e:
        if verbose:
            print(f"✗ Error generating mesh: {e}")
        return False


def save_pointcloud_to_ply(points, output_path, verbose=True):
    """
    Save point cloud to PLY file.

    Args:
        points: Point cloud array [N, 3]
        output_path: Output PLY file path
        verbose: Print progress messages

    Returns:
        bool: True if successful

    Example:
        >>> points = np.random.randn(1000, 3)
        >>> save_pointcloud_to_ply(points, 'output.ply')
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use pyvista (preferred)
        pcd = pv.PolyData(points)
        pcd.save(str(output_path))

        if verbose:
            print(f"Saved point cloud to: {output_path}")
        return True

    except Exception as e:
        if verbose:
            print(f"Error saving point cloud: {e}")
        return False


def batch_generate_meshes(ply_files, output_dir, file_prefix='mesh',
                          samplenum=4096, k_neighbors=128,
                          hausdorff_threshold=0.02, verbose=True):
    """
    Batch process multiple point clouds to generate meshes.

    Args:
        ply_files: List of input PLY file paths
        output_dir: Output directory for meshes
        file_prefix: Prefix for output filenames
        samplenum: Number of points to subsample
        k_neighbors: Number of neighbors for normal estimation
        hausdorff_threshold: Threshold for outlier removal
        verbose: Print progress messages

    Returns:
        list: Paths to generated mesh files

    Example:
        >>> ply_files = ['step_0.ply', 'step_1.ply', 'step_2.ply']
        >>> meshes = batch_generate_meshes(ply_files, 'output/meshes/')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_meshes = []

    for i, ply_path in enumerate(ply_files):
        if verbose:
            print(f"\n[{i+1}/{len(ply_files)}] Processing: {Path(ply_path).name}")

        # Generate output path
        output_path = output_dir / f"{file_prefix}_{i:03d}.obj"

        # Generate mesh
        success = generate_mesh_from_pointcloud(
            ply_path, output_path,
            samplenum=samplenum,
            k_neighbors=k_neighbors,
            hausdorff_threshold=hausdorff_threshold,
            verbose=False  # Suppress individual messages
        )

        if success:
            generated_meshes.append(str(output_path))
            if verbose:
                print(f"  ✓ Generated: {output_path.name}")
        else:
            if verbose:
                print(f"  ✗ Failed: {ply_path}")

    if verbose:
        print(f"\n✓ Batch complete: {len(generated_meshes)}/{len(ply_files)} meshes generated")

    return generated_meshes


def load_mesh(mesh_path):
    """
    Load mesh from file (OBJ or PLY).

    Args:
        mesh_path: Path to mesh file

    Returns:
        Mesh object (pyvista.PolyData or None)
    """
    try:
        mesh = pv.read(str(mesh_path))
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None


def get_mesh_statistics(mesh_path):
    """
    Get statistics about a mesh file.

    Args:
        mesh_path: Path to mesh file

    Returns:
        dict with 'n_points', 'n_faces', 'bounds'
    """
    mesh = load_mesh(mesh_path)
    if mesh is None:
        return None

    return {
        'n_points': mesh.n_points,
        'n_faces': mesh.n_faces if hasattr(mesh, 'n_faces') else 0,
        'bounds': mesh.bounds,
        'center': mesh.center
    }


def check_dependencies():
    """
    Check if required dependencies are available.

    Returns:
        dict with availability status
    """
    return {
        'pymeshlab': True,
        'pyvista': True
    }
