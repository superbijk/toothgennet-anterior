from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import sys
from pathlib import Path
import torch
import numpy as np
import json
import time 
import shutil
import logging
from logging.handlers import RotatingFileHandler

# Add project root to sys.path to allow importing from toothgennet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from toothgennet.sources.utils.model_utils import get_model
from toothgennet.sources.utils.data import get_data_loaders
from toothgennet.sources.visualize.generate import generate_random_samples
from toothgennet.sources.utils.cut_utils import generate_random_cut_condition, split_by_cut_condition

app = Flask(__name__)

# Configure logging to file
LOG_DIR = os.path.abspath(os.path.join(app.root_path, '../../outputs'))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'viewer.log')

# Save original stdout/stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File Handler
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024 * 10, backupCount=5)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Console Handler (writes to original stdout)
console_handler = logging.StreamHandler(original_stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Configure Loggers
def configure_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler) # REMOVED: This causes duplication because StreamToLogger already prints to stdout via logger
    return logger

# Configure specific loggers
app_logger = configure_logger('app') 
werkzeug_logger = configure_logger('werkzeug')
# root_logger = configure_logger(None) # REMOVED: Root logger might be catching things twice

# Add console handler ONLY to werkzeug logger because it doesn't go through sys.stdout redirection
werkzeug_logger.addHandler(console_handler)

# Redirect stdout/stderr to loggers
class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        # Handle bytes if necessary (though usually buf is str for stdout)
        if isinstance(buf, bytes):
            buf = buf.decode('utf-8', errors='replace')
            
        for line in buf.rstrip().splitlines():
            if not line.strip(): continue
            # Log to file
            self.logger.log(self.log_level, line.rstrip())
            # Write to original stdout (so we see it in console)
            # We do this manually here instead of adding a StreamHandler to the logger
            # to avoid the formatting being applied twice or other weirdness
            
            # Ensure we are writing string to stdout (if it expects string)
            # or bytes if it expects bytes. sys.stdout usually expects str in Py3.
            out_line = line.rstrip() + '\n'
            try:
                original_stdout.write(out_line)
            except TypeError:
                # If original_stdout expects bytes (unlikely for sys.stdout but possible if wrapped)
                original_stdout.write(out_line.encode('utf-8'))
            
            original_stdout.flush()

    def flush(self):
        pass

# Redirect stdout/stderr to app.logger
# Since app.logger has a console_handler writing to original_stdout, 
# print() -> sys.stdout -> StreamToLogger -> app.logger -> console_handler -> original_stdout
# This works and adds formatting to print statements.
sys.stdout = StreamToLogger(app.logger, logging.INFO)
sys.stderr = StreamToLogger(app.logger, logging.ERROR)

from toothgennet.sources.utils.model_utils import get_model
from toothgennet.sources.utils.data import get_data_loaders

@app.context_processor
def inject_timestamp():
    return {'TIMESTAMP': time.time()}

# Configuration - read from environment variables
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH',
    "/mnt/c/Users/chawa/OneDrive/CMU_PYTHON/ToothGenNet_Published/session_data/checkpoints/checkpoint-latest.pt")
DATASET_PATH = os.environ.get('DATASET_PATH',
    "/mnt/c/Users/chawa/OneDrive/CMU_PYTHON/ToothGenNet_Published/session_data/dataset/15000_U1.pkl")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables to hold model and data
model = None
dataset = None
data_loader = None

def load_resources():
    global model, dataset, data_loader
    print("Loading resources...")
    model = get_model(CHECKPOINT_PATH, device=DEVICE)
    
    # Load dataset (using validation set for inspection)
    data_loaders = get_data_loaders(DATASET_PATH, tooth_category='U1', n_sample_points=2048)
    data_loader = data_loaders['val_loader']
    # We might want random access, so let's keep the dataset object
    dataset = data_loader.dataset
    print("Resources loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(os.path.join(app.root_path, '../../outputs'), filename)

@app.route('/api/mesh/<int:index>')
def get_mesh(index):
    """Get point cloud data for a specific index."""
    if index >= len(dataset):
        return jsonify({'error': 'Index out of bounds'}), 404
        
    data = dataset[index]
    points = data['sample_points'] # (N, 3)
    
    return jsonify({
        'points': points.tolist(),
        'index': index,
        'category': data['cate_name'],
        'total_samples': len(dataset)
    })

@app.route('/api/clear_generation', methods=['POST'])
def clear_generation():
    try:
        gen_dir = os.path.abspath(os.path.join(app.root_path, '../../outputs/generation'))
        if os.path.exists(gen_dir):
            shutil.rmtree(gen_dir)
        return jsonify({'success': True, 'message': 'Generation cache cleared'})
    except Exception as e:
        print(f"Clear generation failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    import glob
    data = request.json
    num_samples = int(data.get('num_samples', 8))
    regenerate = data.get('regenerate', False)
    save_obj = data.get('save_obj', True)
    
    try:
        base_output_dir = os.path.abspath(os.path.join(app.root_path, '../../outputs/generation'))
        
        output_folder = generate_random_samples(
            checkpoint_path=CHECKPOINT_PATH,
            num_samples=num_samples,
            num_points=15000,
            generate_meshes=save_obj,
            output_dir=base_output_dir,
            regenerate=regenerate
        )
        
        # Construct URLs
        rel_path = os.path.relpath(output_folder, os.path.join(app.root_path, '../../outputs'))
        viz_dir = os.path.join(output_folder, 'visualization')
        
        mesh_files = sorted(glob.glob(os.path.join(viz_dir, 'mesh_*.obj')))
        mesh_urls = [f'/outputs/{rel_path}/visualization/{os.path.basename(f)}' for f in mesh_files]
        
        heatmap_files = sorted(glob.glob(os.path.join(viz_dir, 'latent_interpolation_*.png')))
        heatmap_urls = [f'/outputs/{rel_path}/visualization/{os.path.basename(f)}' for f in heatmap_files]
        
        # Load points
        inferences_dir = os.path.join(output_folder, 'inferences')
        pcd_agg_path = os.path.join(inferences_dir, 'point_clouds.npy')
        point_clouds = []
        if os.path.exists(pcd_agg_path):
            all_points = np.load(pcd_agg_path)
            for pts in all_points:
                point_clouds.append(pts.tolist())

        return jsonify({
            'success': True,
            'mesh_urls': mesh_urls,
            'heatmap_urls': heatmap_urls,
            'point_clouds': point_clouds,
            'message': 'Generation complete'
        })
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_cut', methods=['POST'])
def generate_cut():
    data = request.json
    cut_type = data.get('cut_type', 'split')
    seed = data.get('seed', None)
    
    try:
        cut_formula, cut_params = generate_random_cut_condition(cut_type=cut_type, seed=seed)
        return jsonify({
            'success': True,
            'cut_formula': cut_formula,
            'cut_params': cut_params
        })
    except Exception as e:
        print(f"Cut generation failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview_cut', methods=['POST'])
def preview_cut():
    data = request.json
    sample_idx = int(data.get('sample_idx', 0))
    cut_condition = data.get('cut_condition')
    
    try:
        # Load sample
        if sample_idx >= len(dataset):
            return jsonify({'error': 'Index out of bounds'}), 404
        
        original_points_np = dataset[sample_idx]['sample_points']
        if isinstance(original_points_np, np.ndarray):
            original_points = torch.from_numpy(original_points_np).float()
        else:
            original_points = original_points_np
            
        # Apply cut logic
        # split_by_cut_condition returns (kept_points, excluded_points) as tensors
        kept, excluded = split_by_cut_condition(original_points, cut_condition)
        
        # Concatenate for single buffer visualization
        # Kept first (color 1), Excluded second (color 2 - red)
        combined = torch.cat([kept, excluded], dim=0)
        cut_count = len(kept)
        
        return jsonify({
            'success': True,
            'points': combined.tolist(),
            'cut_count': cut_count
        })
    except Exception as e:
        print(f"Preview cut failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/restore', methods=['POST'])
def restore():
    """
    Run tooth restoration pipeline.
    """
    from toothgennet.sources.visualize.restore_tooth import restore_tooth
    
    data = request.json
    cut_type = data.get('cut_type', 'split')
    cut_condition = data.get('cut_condition') # The formula string
    sample_idx_raw = data.get('sample_idx')
    num_iterations = int(data.get('num_iterations', 200))
    learning_rate = float(data.get('learning_rate', 0.01))
    step_size = int(data.get('step_size', 100))
    gamma = float(data.get('gamma', 0.1))
    viz_steps = int(data.get('viz_steps', 5))
    
    random_sample = False
    sample_idx = 0
    
    if sample_idx_raw is None or sample_idx_raw == '':
        sample_idx = 0 # Default to 0
    else:
        sample_idx = int(sample_idx_raw)
        
    print(f"Running restoration: cut_type={cut_type}, formula={cut_condition}, sample_idx={sample_idx}, iter={num_iterations}, lr={learning_rate}, step={step_size}, gamma={gamma}, viz={viz_steps}")
    
    try:
        base_output_dir = os.path.abspath(os.path.join(app.root_path, '../../outputs/restoration'))
        
        output_folder = restore_tooth(
            checkpoint_path=CHECKPOINT_PATH,
            dataset_path=DATASET_PATH,
            sample_idx=sample_idx,
            cut_type=cut_type,
            cut_formula=cut_condition,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            step_size=step_size,
            gamma=gamma,
            viz_steps=viz_steps,
            output_points=15000,
            random_sample=random_sample,
            output_dir=base_output_dir
        )
        
        # URLs
        rel_path = os.path.relpath(output_folder, os.path.join(app.root_path, '../../outputs'))
        image_urls = [
            f'/outputs/{rel_path}/figures/restoration_steps.png',
            f'/outputs/{rel_path}/figures/optimization_loss.png'
        ]
        
        # Load point clouds for grid visualization
        # 1. Input (kept) points
        input_path = os.path.join(output_folder, 'inputs', 'input_point_cloud.npy')
        input_points = []
        if os.path.exists(input_path):
            input_points = np.load(input_path).tolist()
            
        # 2. Missing (cut) points
        missing_path = os.path.join(output_folder, 'inputs', 'missing_point_cloud.npy')
        missing_points = []
        if os.path.exists(missing_path):
            missing_points = np.load(missing_path).tolist()
            
        # 3. Restored steps (restored parts only)
        # Use the visualization steps (high-res, masked) instead of raw optimization output
        # Updated to use new filename: restored_points_damaged_parts.npy
        steps_path = os.path.join(output_folder, 'inference', 'restored_points_damaged_parts.npy')
        restored_steps = []
        
        # Fallback to old name if new one doesn't exist (for backward compatibility during transition)
        if not os.path.exists(steps_path):
             steps_path = os.path.join(output_folder, 'inference', 'viz_steps_point_clouds.npy')

        if os.path.exists(steps_path):
            # This is [steps, N, 3] or object array
            restored_steps_np = np.load(steps_path, allow_pickle=True)
            print(f"Loaded viz steps from {steps_path}, shape: {restored_steps_np.shape}")
            for i, step in enumerate(restored_steps_np):
                step_list = step.tolist()
                restored_steps.append(step_list)
                if i == 0:
                    print(f"Step 0 points: {len(step_list)}")
        else:
            # Fallback to raw output if viz steps not found (legacy support)
            # Updated to check both old and new names for raw output
            steps_path = os.path.join(output_folder, 'inference', 'restored_points_from_optimized_latents.npy')
            if not os.path.exists(steps_path):
                steps_path = os.path.join(output_folder, 'inference', 'restored_point_clouds.npy')
                
            if os.path.exists(steps_path):
                restored_steps_np = np.load(steps_path)
                # Subsample if too many
                if len(restored_steps_np) > viz_steps:
                    indices = np.linspace(0, len(restored_steps_np)-1, viz_steps, dtype=int)
                    restored_steps_np = restored_steps_np[indices]
                
                for step in restored_steps_np:
                    restored_steps.append(step.tolist())

        print(f"Sending response: {len(restored_steps)} steps, {len(input_points)} input points")
        return jsonify({
            'success': True,
            'image_urls': image_urls,
            'input_points': input_points,
            'missing_points': missing_points,
            'restored_steps': restored_steps,
            'message': 'Restoration complete'
        })
        
    except Exception as e:
        print(f"Restoration failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/interpolate', methods=['POST'])
def interpolate():
    """Interpolate between two latent vectors."""
    from toothgennet.sources.visualize.interpolate import interpolate_enhanced
    import glob
    
    data = request.json
    idx1 = data.get('idx1')
    idx2 = data.get('idx2')
    steps = data.get('steps', 10)
    regenerate = data.get('regenerate', False)
    
    # Handle random indices
    if idx1 is None or idx1 == '': idx1 = -1
    else: idx1 = int(idx1)
    
    if idx2 is None or idx2 == '': idx2 = -1
    else: idx2 = int(idx2)
    
    print(f"Running interpolation: {idx1} -> {idx2}, steps={steps}, regenerate={regenerate}")
    
    try:
        base_output_dir = os.path.abspath(os.path.join(app.root_path, '../../outputs/latent_interpolation'))
        
        output_folder = interpolate_enhanced(
            checkpoint_path=CHECKPOINT_PATH,
            dataset_path=DATASET_PATH,
            idx1=idx1,
            idx2=idx2,
            steps=steps,
            num_points=15000,
            generate_meshes=True,
            output_dir=base_output_dir,
            regenerate=regenerate
        )
        
        # URLs
        rel_path = os.path.relpath(output_folder, os.path.join(app.root_path, '../../outputs'))
        
        viz_dir = os.path.join(output_folder, 'visualization')
        heatmap_urls = []
        mesh_urls = []
        
        if os.path.exists(viz_dir):
             mesh_files = sorted(glob.glob(os.path.join(viz_dir, 'mesh_*.obj')))
             mesh_urls = [f'/outputs/{rel_path}/visualization/{os.path.basename(f)}' for f in mesh_files]
             
             heatmap_files = sorted(glob.glob(os.path.join(viz_dir, 'latent_interpolation_*.png')))
             heatmap_urls = [f'/outputs/{rel_path}/visualization/{os.path.basename(f)}' for f in heatmap_files]
        
        inferences_dir = os.path.join(output_folder, 'inferences')
        pcd_agg_path = os.path.join(inferences_dir, 'point_clouds.npy')
        
        point_clouds = []
        if os.path.exists(pcd_agg_path):
            all_points = np.load(pcd_agg_path)
            for pts in all_points:
                 point_clouds.append(pts.tolist())

        return jsonify({
            'success': True,
            'mesh_urls': mesh_urls,
            'heatmap_urls': heatmap_urls,
            'point_clouds': point_clouds,
            'message': 'Interpolation complete'
        })
        
    except Exception as e:
        print(f"Interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        load_resources()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        app.logger.error(f"Failed to start application: {e}", exc_info=True)
        raise