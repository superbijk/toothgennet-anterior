# ToothGenNet Modules

This directory contains external dependencies required for the project.

## Dependencies

1.  **PointFlow**:
    ```bash
    git clone https://github.com/stevenygd/PointFlow.git
    ```
    *Note: Ensure the `PointFlow` directory is placed here.*

2.  **PyTorch PCD Metrics**:
    ```bash
    git clone https://github.com/superbijk/pytorch_pcd_metrics.git
    ```
    *Note: Ensure the `pytorch_pcd_metrics` directory is placed here.*

## Installation

After cloning, you may need to compile the CUDA kernels for `pytorch_pcd_metrics` if you want to use the optimized versions. Refer to the respective repositories for detailed installation instructions.
