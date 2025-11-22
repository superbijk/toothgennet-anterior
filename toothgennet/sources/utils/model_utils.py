import argparse
import torch
import sys
from pathlib import Path

# Add modules to path
modules_path = Path(__file__).resolve().parents[2] / "modules"
sys.path.insert(0, str(modules_path))

from PointFlow.models.networks import PointFlow

class PointFlowArgs(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__()
        # Default Model args
        self.input_dim = 3
        self.zdim = 128
        self.dims = '512-512-512'
        self.latent_dims = '256-256'
        self.num_blocks = 1
        self.latent_num_blocks = 1
        self.use_latent_flow = True
        self.use_deterministic_encoder = False
        self.n_sample_points = 2048
        # Flow args
        self.layer_type = 'concatsquash'
        self.nonlinearity = 'tanh'
        self.batch_norm = True
        self.time_length = 0.5
        self.train_T = True
        self.use_adjoint = True
        self.solver = 'dopri5'
        self.atol = 1e-5
        self.rtol = 1e-5
        self.bn_lag = 0
        self.sync_bn = False
        # weight and optimizer / scheduler args
        self.prior_weight = 1.0
        self.recon_weight = 1.0
        self.entropy_weight = 1.0
        self.lr = 2e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.momentum = 0.9
        self.weight_decay = 0.0
        self.scheduler = 'linear'
        self.exp_decay = 1.0
        self.exp_decay_freq = 1
        # Training args
        self.epochs = 1000
        self.distributed = False
        self.batch_size = 16
        self.test_batch_size = 32
        # Saving args
        self.step_log_freq = 1
        self.epoch_save_freq = 2
        self.epoch_viz_freq = 1

        # Override defaults with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def get_model(checkpoint_path=None, device='cpu', **kwargs):
    """
    Initializes and loads the PointFlow model.

    Follows Teeth_CVAE pattern: build model, move to device, then load checkpoint.

    Args:
        checkpoint_path (str or Path, optional): Path to the model checkpoint.
        device (str or torch.device): Device to load the model on.
        **kwargs: Arguments to override default PointFlowArgs.

    Returns:
        model (PointFlow): The initialized model.
    """
    # Convert device to torch.device if string
    if isinstance(device, str):
        device = torch.device(device)

    args = PointFlowArgs(**kwargs)
    model = PointFlow(args)

    # Move model to device FIRST (before loading checkpoint)
    # Use .cuda() for CUDA devices to ensure all buffers are moved correctly
    # PointFlow's sample_gaussian uses .cuda(gpu) internally, so we must ensure
    # all model components (including BatchNorm buffers) are on CUDA
    if device.type == 'cuda':
        model = model.cuda(device.index if device.index is not None else 0)
    else:
        model = model.to(device)

    if checkpoint_path:
        # Load checkpoint with weights_only=False (same as Teeth_CVAE)
        # This ensures all buffers and parameters load correctly
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle both full checkpoint dict and state_dict only
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    return model
