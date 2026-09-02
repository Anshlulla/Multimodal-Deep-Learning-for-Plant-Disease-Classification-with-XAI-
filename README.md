# Plant Disease Classification Project

A comprehensive machine learning pipeline for automated plant disease detection and classification using advanced deep learning techniques.

## Project Overview

This project implements a multi-faceted approach to plant disease detection combining:
- **Morphological Segmentation** - Automated leaf isolation using computer vision
- **Transfer Learning** - Pre-trained models (MobileNetV2, EfficientNetB0, ResNet50)
- **U-Net Architecture** - Advanced semantic segmentation
- **Multimodal Learning** - Integration of visual features with text descriptions using vision-language models

## Dataset

The project works with a plant disease dataset containing **22 disease classes** across multiple plant species:
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Corn (Maize)**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Pepper (Bell)**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Tomato**: Target Spot, Tomato mosaic virus, Yellow Leaf Curl Virus, Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (Two-spotted), Healthy

### Data Organization

```
Dataset/                          # Original dataset (images by class)
Dataset_Split/                    # Train/Val/Test split (prevents data leakage)
Dataset_Split_Morphological/      # Segmented images (leaf isolation)
Dataset_Split_UNet/               # U-Net segmentation outputs
outputs/                          # Generated configs and weights
  ├── class_names.json           # Class mapping
  ├── class_weights.json         # Balanced class weights
  └── dataset_config.json        # Dataset metadata
```

## Notebook Workflow

Execute notebooks in the following sequential order:

### Phase 1: Data Preparation & Validation

#### **01_eda_leakage_check.ipynb**
**Purpose**: Exploratory Data Analysis and Data Leakage Detection
- Loads and analyzes dataset structure
- Counts images per class
- Visualizes class distribution
- Identifies duplicate/augmented images
- Detects potential data leakage (same image in multiple splits)
- **Output**: EDA visualizations and leakage report

#### **02_fix_leakage_resplit.ipynb**
**Purpose**: Fix Data Leakage by Repartitioning at Original ID Level
- Extracts original image IDs and groups augmentations
- Ensures all augmentations of the same original image stay together
- Splits dataset into Train/Val/Test (70/15/15) at original ID level
- Creates `Dataset_Split/` directory with clean, leakage-free splits
- **Output**: `Dataset_Split/` with train/val/test subdirectories

### Phase 2: Segmentation & Preprocessing

#### **03_segmentation_morphological.ipynb**
**Purpose**: Morphological Leaf Segmentation
- Implements HSV color-space based leaf detection
- Uses morphological operations (erosion, dilation, closing)
- Extracts largest contour (leaf) from each image
- Applies segmentation mask to isolate leaves
- Batch processes entire dataset
- **Output**: `Dataset_Split_Morphological/` with segmented images

#### **04_class_weights.ipynb**
**Purpose**: Compute Balanced Class Weights
- Counts images per class in morphological dataset
- Computes sklearn balanced class weights
- Addresses class imbalance for training
- Saves weights to `outputs/class_weights.pkl`
- Saves class names to `outputs/class_names.json`
- **Output**: JSON/pickle files with class mappings and weights

#### **05_preprocessing_pipeline.ipynb**
**Purpose**: TensorFlow Data Pipeline Setup
- Loads class names and weights from outputs
- Builds efficient tf.data preprocessing pipeline
- Implements image resizing (224×224)
- Configures data augmentation (random flip, rotation, zoom)
- Sets up batching and prefetching
- **Output**: Reusable data pipeline functions

### Phase 3: Model Training

#### **06_transfer_learning_morphological.ipynb**
**Purpose**: Transfer Learning on Morphologically Segmented Data
- Uses pre-trained feature extractors: MobileNetV2, EfficientNetB0
- Trains classification head on segmented images
- Implements custom callbacks (checkpoint saving every N epochs)
- Uses balanced class weights
- Tracks training/validation metrics
- **Output**: Trained models saved to `outputs/models_morphological/`

#### **07_segmentation_transfer_learning_unet.ipynb**
**Purpose**: U-Net Architecture for Advanced Segmentation
- Implements full U-Net encoder-decoder architecture
- Trains on morphological dataset (supervised segmentation)
- Generates binary masks for leaf regions
- Applies U-Net segmentation to entire dataset
- **Output**: `Dataset_Split_UNet/` with U-Net segmentation masks

#### **08_transfer_learning_original.ipynb**
**Purpose**: Transfer Learning on Original (Unsegmented) Images
- Trains on original Dataset_Split without morphological preprocessing
- Compares transfer learning performance baseline
- Tests models: MobileNetV2, EfficientNetB0, ResNet50
- Includes training history and metrics evaluation
- Generates confusion matrices and classification reports
- **Output**: Models and performance comparisons

#### **09_multimodal_training.ipynb**
**Purpose**: Multimodal Plant Disease Classification
- Integrates vision and language modalities
- Uses LLaVA (vision-language model) for caption generation
- Generates descriptive text for each disease image
- Combines visual features (ResNet50) with text embeddings
- Trains multimodal fusion model
- Improves interpretability and performance
- **Output**: Multimodal model, captions, and fused representations

## Key Features

### Data Quality
- ✅ Automated leakage detection and prevention
- ✅ Augmentation-aware train/val/test splitting
- ✅ Balanced class weight computation
- ✅ Image preprocessing and normalization

### Computer Vision Techniques
- 🔬 Morphological operations (erosion, dilation, closing)
- 🔬 HSV color-space segmentation
- 🔬 Contour detection and extraction
- 🔬 U-Net deep segmentation

### Deep Learning Models
- 🤖 MobileNetV2 (lightweight, efficient)
- 🤖 EfficientNetB0 (optimized accuracy-efficiency tradeoff)
- 🤖 ResNet50 (robust feature extraction)
- 🤖 U-Net (semantic segmentation)
- 🤖 LLaVA (multimodal vision-language)

### Training Strategies
- 📊 Transfer learning with fine-tuning
- 📊 Balanced class weights
- 📊 Data augmentation (flip, rotation, zoom)
- 📊 Custom callbacks and checkpointing
- 📊 Multimodal learning fusion

## Requirements

### Dependencies
```
tensorflow>=2.12.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.66.0
```

### Hardware Recommendations
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: Multi-core processor for preprocessing
- **RAM**: 16GB+ for full dataset processing
- **Storage**: 50GB+ for dataset and models

## Installation

1. **Clone the repository**
   ```bash
   cd c:\Users\Ansh Lulla\VS-Code\PlantDiseaseProject\plant-disease-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Jupyter**
   ```bash
   pip install jupyter
   jupyter notebook
   ```

## Usage

### Running the Complete Pipeline

```bash
# Activate environment
.venv\Scripts\Activate.ps1

# Start Jupyter
jupyter notebook

# Navigate to src/notebooks/ and run in order:
# 1. 01_eda_leakage_check.ipynb
# 2. 02_fix_leakage_resplit.ipynb
# 3. 03_segmentation_morphological.ipynb
# 4. 04_class_weights.ipynb
# 5. 05_preprocessing_pipeline.ipynb
# 6. 06_transfer_learning_morphological.ipynb (or choose alternative models)
# 7. 07_segmentation_transfer_learning_unet.ipynb
# 8. 08_transfer_learning_original.ipynb
# 9. 09_multimodal_training.ipynb
```

### Running Individual Notebooks

Each notebook is self-contained with configuration sections at the top for:
- Dataset paths
- Model hyperparameters
- Training settings
- Output directories

Modify these sections as needed for your environment.

## Output Directory Structure

After running all notebooks:
```
outputs/
├── class_names.json              # 22 plant disease classes
├── class_weights.pkl             # Balanced weights
├── dataset_config.json           # Dataset statistics
├── models_morphological/         # Transfer learning on segmented data
│   ├── mobilenetv2_checkpoint_*.h5
│   └── efficientnetb0_checkpoint_*.h5
├── models_unet/                  # U-Net segmentation models
│   └── unet_model_*.h5
├── models_multimodal/            # Multimodal fusion models
│   └── multimodal_model.h5
└── captions/                     # Generated image captions
    └── captions.json
```

## Model Performance Considerations

### Morphological Segmentation Approach
- **Pros**: Improved leaf isolation, noise reduction
- **Cons**: May fail on certain lighting conditions or disease types
- **Best for**: Well-lit images with clear leaf boundaries

### Transfer Learning
- **Lightweight** (MobileNetV2): Fast inference, lower accuracy
- **Balanced** (EfficientNetB0): Good accuracy-efficiency tradeoff
- **Robust** (ResNet50): Higher accuracy, more compute

### U-Net Segmentation
- **Advanced**: Learns pixel-level segmentation
- **Flexible**: Works across various image conditions
- **Overhead**: Requires labeled masks for training

### Multimodal Learning
- **Interpretable**: Text descriptions explain predictions
- **Enhanced**: Combines visual and semantic information
- **Complex**: Requires caption generation and text encoding

## Troubleshooting

### Data Leakage Detected
- ✓ Run notebook 02 to fix leakage through ID-level splitting

### Out of Memory Errors
- Reduce `BATCH_SIZE` in config
- Reduce `SUBSET_PERCENTAGE` for quick tests
- Process smaller dataset splits

### Poor Model Performance
- Verify data split is leakage-free
- Check class weight computation
- Tune learning rate and epochs
- Ensure balanced class distribution

### Missing Output Files
- Verify paths in notebook config sections
- Ensure previous notebooks completed successfully
- Check disk space for output directory

## Project Structure

```
src/
├── notebooks/
│   ├── 01_eda_leakage_check.ipynb
│   ├── 02_fix_leakage_resplit.ipynb
│   ├── 03_segmentation_morphological.ipynb
│   ├── 04_class_weights.ipynb
│   ├── 05_preprocessing_pipeline.ipynb
│   ├── 06_transfer_learning_morphological.ipynb
│   ├── 07_segmentation_transfer_learning_unet.ipynb
│   ├── 08_transfer_learning_original.ipynb
│   ├── 09_multimodal_training.ipynb
│   └── outputs/          # Temporary notebook outputs
├── pyproject.toml        # Project metadata
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Key Insights

1. **Data Leakage Prevention**: The pipeline ensures augmented images stay together during splitting
2. **Segmentation as Preprocessing**: Morphological segmentation improves classification by isolating target objects
3. **Multiple Approaches**: Tests different architectures and preprocessing strategies
4. **Multimodal Fusion**: Combines visual and textual representations for better interpretability
5. **Production Ready**: Balanced weights, proper data pipelines, and checkpoint management

## Performance Tips

- Use morphological segmentation for structured datasets with clear leaf boundaries
- Use transfer learning for rapid prototyping with limited data
- Use U-Net for more challenging segmentation scenarios
- Use multimodal learning when interpretability is important
- Start with MobileNetV2 for quick iterations, then scale to EfficientNetB0 or ResNet50

## Future Enhancements

- [ ] Ensemble methods combining multiple model predictions
- [ ] Real-time inference pipeline
- [ ] Mobile app deployment (TFLite conversion)
- [ ] Advanced augmentation techniques (mixup, cutmix)
- [ ] Explainable AI (GradCAM visualizations)
- [ ] Few-shot learning for new disease detection
- [ ] Attention mechanisms for interpretability

## References

### Papers
- MobileNetV2: Sandler et al. (2018)
- EfficientNet: Tan & Le (2019)
- U-Net: Ronneberger et al. (2015)
- LLaVA: Liu et al. (2023)

### Libraries
- TensorFlow/Keras: Deep learning framework
- OpenCV: Computer vision operations
- scikit-learn: Machine learning utilities

## License

This project is provided for educational and research purposes.
---

**Last Updated**: 2026-09-02  
**Python Version**: 3.8+  
**TensorFlow Version**: 2.12.0+
