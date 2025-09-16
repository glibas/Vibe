# 🎨 Vibe: Vision transformer for Interface Beauty Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glibas/Vibe/blob/main/notebooks/QuickStart.ipynb)

**VIBE** (**V**ision transformer for **I**nterface **B**eauty **E**valuation) is a comprehensive deep learning framework for **predicting web interface aesthetics using saliency-guided Vision Transformers with multi-scale ROI selection**. 

Built with modular components optimized for Google Colab, Vibe leverages advanced techniques including:
- 🎯 **Charm-enhanced patch selection** with saliency guidance
- 🔍 **SUM-based saliency map generation** for attention focus
- 📐 **Web ROI (Region of Interest) detection** for interface components
- 🧠 **Multi-scale Vision Transformers** for comprehensive beauty assessment

## 🌟 Features

- **🎯 Advanced Patch Selection**:
  - [Charm approach](https://github.com/FBehrad/Charm) enhanced with saliency guidance
  - Intelligent patch sampling for optimal feature extraction
  - Multi-scale analysis for comprehensive interface understanding

- **🔍 Saliency-Guided Processing**:
  - [SUM (Saliency for User's Attention)](https://github.com/Arhosseini77/SUM) integration
  - Visual attention maps to focus on interface hotspots
  - Web ROI enhancement for interface-specific regions

- **🧠 Multiple Model Architectures**:
  - Basic Vision Transformer for interface analysis
  - Saliency-guided ViT with attention to visual importance
  - ROI-enhanced ViT with multi-scale region selection

- **📊 Comprehensive Beauty Analysis**:
  - Overall beauty score prediction (0-1 scale)
  - Beauty aspects evaluation (color, layout, typography, balance)
  - Design principles assessment (symmetry, contrast, hierarchy, alignment, whitespace)

- **🛠️ Google Colab Optimized**:
  - Modular notebook components for easy experimentation
  - GPU-accelerated training and inference
  - Seamless Google Drive dataset integration
  - Pre-configured environments for immediate use

## 🚀 Quick Start

### 📓 Google Colab (Recommended)

Get started immediately with our pre-configured notebooks:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glibas/Vibe/blob/main/notebooks/QuickStart.ipynb)

```python
# In Google Colab
!git clone https://github.com/glibas/Vibe.git
%cd Vibe
!pip install -q -r requirements.txt
```

**Dataset Setup in Colab:**
```python
from google.colab import drive
import shutil, zipfile, os

# Mount Google Drive
drive.mount('/content/drive')

# Load WebDesignPrototypicality dataset
zip_path = '/content/drive/MyDrive/datasets/webdesignprototypicality.zip'
local_zip = '/content/webdesignprototypicality.zip'
extract_dir = '/content/webdesignprototypicality'

if not os.path.exists(extract_dir):
    shutil.copy(zip_path, local_zip)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Data extracted to:", extract_dir)
```

### 💻 Local Installation

```bash
# Clone the repository
git clone https://github.com/glibas/Vibe.git
cd Vibe

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

1. **Create sample data for testing**:
```bash
vibe-create-sample-data --save-dir ./sample_data --num-samples 100
```

2. **Train a model**:
```bash
vibe-train --data-path ./sample_data --annotations ./sample_data/annotations.csv --output-dir ./outputs --epochs 50
```

3. **Evaluate a trained model**:
```bash
vibe-eval --model-path ./outputs/checkpoints/best.pth --data-path ./sample_data --annotations ./sample_data/annotations.csv --output-dir ./evaluation
```

4. **Predict beauty score for a single image**:
```bash
vibe-predict --image-path ./sample_data/images/interface_001.png --model-path ./outputs/checkpoints/best.pth --output-dir ./prediction
```

## 📖 Documentation

### Model Architectures

#### 1. Basic Vision Transformer (`basic`)
- Standard ViT architecture adapted for interface analysis
- Global beauty score prediction with beauty-specific tokens
- Suitable for straightforward beauty evaluation tasks

#### 2. Saliency-Guided Vision Transformer (`saliency_guided`)
- Incorporates saliency maps to guide attention
- Focuses on visually important regions in interfaces
- Enhanced performance on complex interface layouts

#### 3. ROI-Enhanced Vision Transformer (`roi_enhanced`)
- Multi-scale region of interest detection
- Combines global and local feature analysis
- Best performance for detailed interface assessment

### Data Format

Vibe supports multiple dataset formats, with primary support for the **WebDesignPrototypicality** dataset:

**📖 Supported Datasets:**
- [WebDesignPrototypicality Dataset](https://www.sciencedirect.com/science/article/pii/S2352340923010077) - Primary dataset for training and evaluation
- Custom interface datasets with beauty annotations

**📁 Expected Directory Structure:**
```
data/
├── images/
│   ├── interface_001.png
│   ├── interface_002.png
│   └── ...
└── annotations.csv
```

**📊 Annotations CSV Format:**
```csv
image_path,beauty_score,color_score,layout_score,typography_score,balance_score,symmetry_score,contrast_score,hierarchy_score,alignment_score,whitespace_score
images/interface_001.png,0.85,0.8,0.9,0.7,0.8,0.75,0.85,0.8,0.9,0.7
images/interface_002.png,0.62,0.6,0.7,0.5,0.6,0.65,0.7,0.6,0.7,0.5
```

**🔄 Google Colab Dataset Loading:**
```python
from vibe.data.colab_utils import load_webdesign_dataset

# Automatic dataset setup in Colab
dataset_path = load_webdesign_dataset(
    drive_path='/content/drive/MyDrive/datasets/webdesignprototypicality.zip',
    extract_to='/content/webdesignprototypicality'
)
```

### Configuration

Use YAML configuration files for complex training setups:

```yaml
# config/custom.yaml
model:
  model_type: "saliency_guided"
  img_size: 224
  embed_dim: 768
  n_layers: 12
  n_heads: 12
  use_beauty_tokens: true
  saliency_weight: 0.5

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"

data:
  train_data_path: "./data/train"
  train_annotations: "./data/train_annotations.csv"
  val_data_path: "./data/val"
  val_annotations: "./data/val_annotations.csv"
```

Train with configuration:
```bash
vibe-train --config ./config/custom.yaml
```

### Python API

```python
import torch
from vibe.models import BeautyPredictor
from vibe.data import InterfaceDataLoader
from vibe.training import BeautyTrainer

# Create model
model = BeautyPredictor(
    model_type='saliency_guided',
    img_size=224,
    embed_dim=768,
    use_beauty_tokens=True
)

# Load data
train_loader = InterfaceDataLoader.create_train_loader(
    data_path='./data',
    annotations_file='./annotations.csv',
    batch_size=32
)

# Train model
trainer = BeautyTrainer(model, train_loader, val_loader, optimizer)
trainer.train(num_epochs=100)

# Analyze interface
analysis = model.analyze_interface(image_tensor)
beauty_score = analysis['overall_beauty_score']
explanation = model.get_beauty_explanation(analysis)
```

## 🔧 Advanced Usage

### Custom Model Training

```python
from vibe.models import BeautyPredictor
from vibe.training import BeautyTrainer, BeautyLoss
from vibe.config import ConfigManager

# Load configuration
config = ConfigManager.load_config('config/advanced.yaml')
model_config, training_config, data_config, exp_config = (
    ConfigManager.create_configs_from_dict(config)
)

# Create model with custom parameters
model = BeautyPredictor(
    model_type='roi_enhanced',
    roi_scales=[32, 64, 128, 256],
    roi_fusion_method='attention',
    saliency_weight=0.7
)

# Custom loss function
loss_fn = BeautyLoss(
    beauty_weight=1.0,
    aspect_weight=0.6,
    principle_weight=0.4,
    consistency_weight=0.3
)

# Advanced training
trainer = BeautyTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn
)
```

### Model Evaluation and Analysis

```python
from vibe.evaluation import InterfaceEvaluator

# Comprehensive evaluation
evaluator = InterfaceEvaluator(model, device='cuda')
results = evaluator.evaluate_dataset(test_loader, save_dir='./evaluation')

# Detailed analysis
analyses = evaluator.analyze_predictions(test_loader, n_samples=20)

# Generate report
report = evaluator.get_summary_report()
print(report)
```

### Visualization and Interpretation

```python
# Get attention maps
attention_maps = model.get_attention_maps(image_tensor, layer_idx=-1)

# Get saliency visualization
saliency_viz = model.get_saliency_visualization(image_tensor)

# Comprehensive interface analysis
analysis = model.analyze_interface(image_tensor)
explanation = model.get_beauty_explanation(analysis)

print(f"Beauty Score: {analysis['overall_beauty_score']:.3f}")
print(f"Beauty Aspects: {analysis['beauty_aspects']}")
print(f"Design Principles: {analysis['design_principles']}")
print(f"\nExplanation:\n{explanation}")
```

## 📚 Google Colab Notebooks

Vibe includes comprehensive Jupyter notebooks optimized for Google Colab:

| Notebook | Description | Link |
|----------|-------------|------|
| **QuickStart.ipynb** | Get started with Vibe in under 5 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glibas/Vibe/blob/main/notebooks/QuickStart.ipynb) |
| **TrainingNotebook.ipynb** | Complete training pipeline with GPU acceleration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glibas/Vibe/blob/main/notebooks/TrainingNotebook.ipynb) |
| **EvaluationNotebook.ipynb** | Comprehensive model evaluation and metrics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glibas/Vibe/blob/main/notebooks/EvaluationNotebook.ipynb) |

### 🗂️ Modular Components

Each notebook uses modular Vibe components that can be easily adapted:

```python
# Dataset loading module
from vibe.data.colab_utils import load_webdesign_dataset, quick_setup

# Model creation module  
from vibe.models import BeautyPredictor

# Training module
from vibe.training import BeautyTrainer, BeautyLoss

# Evaluation module
from vibe.evaluation import InterfaceEvaluator
```

## 📊 Evaluation Metrics

Vibe provides comprehensive evaluation metrics:

- **Regression Metrics**: RMSE, MAE, Pearson correlation, Spearman correlation
- **Beauty Aspects**: Individual assessment of color, layout, typography, balance
- **Design Principles**: Evaluation of symmetry, contrast, hierarchy, alignment, whitespace
- **Consistency**: Alignment between overall beauty and individual aspects

## 🛠️ Development

### Project Structure

```
Vibe/
├── vibe/                    # Main package
│   ├── models/             # Model architectures
│   │   ├── vision_transformer.py
│   │   ├── saliency_guided_vit.py
│   │   ├── roi_selector.py
│   │   └── beauty_predictor.py
│   ├── data/               # Data processing
│   ├── training/           # Training utilities
│   ├── evaluation/         # Evaluation tools
│   ├── utils/              # Utility functions
│   ├── config.py           # Configuration management
│   └── cli.py              # Command line interface
├── examples/               # Example scripts
├── config/                 # Configuration files
├── requirements.txt        # Dependencies
└── setup.py               # Package setup
```

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=vibe
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📝 Citation

If you use Vibe in your research, please cite:

```bibtex
@misc{vibe2024,
  title={Vibe: Vision Transformer for Interface Beauty Evaluation},
  author={Vibe Team},
  year={2024},
  howpublished={\url{https://github.com/glibas/Vibe}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- **Vision Transformer**: Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **Charm Approach**: Enhanced patch selection using [FBehrad/Charm](https://github.com/FBehrad/Charm)
- **SUM Saliency**: Saliency maps generated using [Arhosseini77/SUM](https://github.com/Arhosseini77/SUM)
- **WebDesignPrototypicality Dataset**: [Todi et al. 2023](https://www.sciencedirect.com/science/article/pii/S2352340923010077)
- **Interface beauty evaluation**: Metrics from HCI and design research
- **ROI selection**: Techniques adapted from object detection and web analysis literature

## 📞 Contact

- GitHub Issues: [https://github.com/glibas/Vibe/issues](https://github.com/glibas/Vibe/issues)
- Email: contact@vibe-ai.com

---

**Vibe** - Making interface beauty evaluation as simple as a single line of code! 🎨✨
