# Plastinka Photoassistant

A tool for batch processing photos of vinyl record sleeves and booklets. It includes a Streamlit web interface with authentication and a modular image-processing pipeline: segmentation (YOLOv8), mask post-processing and framing, exposure/balance correction using ArUco, perspective rectification, and resizing.

## Features
- **Web UI**: upload images, track progress, download results as a ZIP
- **Authentication**: user management via `streamlit-authenticator`
- **Modular pipeline**: easily configurable module order and parameters via YAML
- **YOLOv8 segmentation**: target object and mask detection with CPU/GPU support
- **Mask post-processing**: contour approximation, beveled-corner restoration, border construction
- **Exposure/balance correction**: based on a white patch near ArUco marker
- **Perspective rectification**: warp to square/rectangular target format
- **Resize**: scale to the desired longest side
- **Logs and reports**: store processing errors and summary reports

## Project structure
```
photoassist/                 # Package with pipeline and modules
  modules/
    base_module.py           # Base class for pipeline modules
    segmenter.py             # YOLOv8 segmentation
    mask_post_processing.py  # Post-processing of mask and border computation
    balancer.py              # Balance correction using ArUco-based white patch
    perspective_warper.py    # Perspective rectification
    framer.py                # Frame composition using mask/class
    resizer.py               # Final image resizing
  pipeline/
    pipeline.py              # Pipeline orchestration, parallelization
    config.py                # YAML configuration wrapper

user_interface/
  app.py                     # Streamlit pages configuration
  image_processing.py        # Main processing UI and download
  account_management.py      # User management
  my_logging.py              # Logging setup

main.py                      # Streamlit entry point (navigation.run())
pyproject.toml               # Project configuration and dependencies
uv.lock                      # Lock file for reproducible builds
.python-version              # Python version specification
default_config.yaml          # Default pipeline configuration
debug_config.yaml            # Debug configuration (with intermediate outputs)
```

## Requirements
- Python 3.11+
- uv (Python package manager)
- PyTorch (CPU or CUDA as desired)
- OpenCV with ArUco modules (`opencv-contrib-python`)
- Ultralytics (YOLOv8)

## Installation

### Prerequisites
1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Ensure Python 3.11+ is available

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd plastinka-photoassistant

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
uv run --help  # This will show available commands
```

> Note: for GPU support, install the appropriate `torch` build per the PyTorch instructions. The project uses `torch==2.3.1` by default.

### PyTorch Backend Automation
uv can automatically detect and select the appropriate PyTorch backend for your system:

```bash
# Automatic backend detection (recommended)
uv pip install torch --torch-backend=auto

# Or set environment variable for all commands
export UV_TORCH_BACKEND=auto
uv sync
```

Available backends:
- `auto` - Automatically detects CUDA, AMD, Intel GPU, or defaults to CPU
- `cpu` - CPU-only builds
- `cu126` - CUDA 12.6
- `cu121` - CUDA 12.1
- `cu118` - CUDA 11.8
- `xpu` - Intel XPU
- `rocm` - AMD ROCm

You can also configure this in `pyproject.toml`:
```toml
[tool.uv.pip]
torch-backend = "auto"
```

### Useful uv Commands
```bash
# Install dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv lock --upgrade

# Run any command in the virtual environment
uv run python script.py

# Activate the virtual environment (if needed)
uv shell

# Install PyTorch with automatic backend detection
uv pip install torch --torch-backend=auto

# Install with specific backend
uv pip install torch --torch-backend=cu126

# Install optional dependencies
uv sync --extra cpu
uv sync --extra cu126
```

## Model weights
By default the model path is taken from `default_config.yaml`:
```yaml
modules:
  Segmenter:
    model: yolo_v8_s_800_1.pt
```
Copy `yolo_v8_s_800_1.pt` to the project root or set an absolute/relative path in the configuration.

## Configuration
The pipeline is configured via YAML (`default_config.yaml`, `debug_config.yaml`). Example:
```yaml
modules:
  Segmenter:           # module init order and parameters
    order: 0
    conf_threshold: 0.5
    model: yolo_v8_s_800_1.pt
    device: cpu            # or cuda:0
    save_intermediate_outputs: False
  Balancer:
    order: 1
    aruco_dict: DICT_4X4_50
    aruco_idx: 0
    offset: 10
    C: 1
    save_intermediate_outputs: False
  PostProcessor:
    order: 3
    max_dist: 150
    min_angle: 90
    ...
  PerspectiveWarper:
    order: 4
    interpolation: INTER_CUBIC
  Resizer:
    order: 5
    longest_side: 1500

pipeline:
  n_jobs: 1                 # number of parallel jobs for joblib
  save_intermediate_outputs: False
```
- `order`: execution order of modules
- `save_intermediate_outputs`: when True the module adds a visualization of its step (shown in the UI, useful for debugging)
- Other parameters are module-specific (see code in `photoassist/modules/*`).

## Secrets and authentication
Uses `streamlit-authenticator`. Before running, create `st_secrets.yaml` in the project root:
```yaml
credentials:
  usernames:
    admin:
      email: admin@example.com
      name: Administrator
      password: "hashed_password"
pre-authorized:
  emails: []
cookie:
  name: some_cookie
  key: some_key
  expiry_days: 30
```
- You can create/update users on the “User Management” page. The `st_secrets.yaml` file will be rewritten.

## Run
```bash
# Run the Streamlit application
uv run streamlit run main.py
```

- You will see a multi-page UI:
  - "Image Processing": upload files, progress, download archive
  - "User Management": register/edit/delete users
  - "Errors": admin-only, view logs from `logs/errors`

## Programmatic usage
```python
from PIL import Image
from photoassist import PipelineConfig, Pipeline

config = PipelineConfig('default_config.yaml')
pipeline = Pipeline(config)

inputs = [
  {"image": Image.open("in1.jpg"), "name": "in1.jpg"},
  {"image": Image.open("in2.jpg"), "name": "in2.jpg"},
]
results = pipeline(inputs)
# Each item is a dict with keys 'image' (np.ndarray BGR) and 'name'
```

## Logs and reports
- Step errors are stored in `logs/errors` (see `user_interface/my_logging.py`)

## Debugging
Use `debug_config.yaml` which enables intermediate outputs for all modules. They are shown in the UI for the current image.

## License
AGPL3.0