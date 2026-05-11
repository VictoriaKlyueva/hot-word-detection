# Hot Word Detection

This project is a hot word detection system that identifies the word "stones" in audio files and radio streams using a CNN model with MFCC features.

## The Data Science Part

### Steps
- Data extraction and processing
- Data augmentation (noise, pitch, time stretch, filters)
- Model architecture design
- Model training
- Model evaluating
- Production-ready CLI implementation

### Training

#### Parameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Epochs | 50 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Scheduler | CosineAnnealingWarmRestarts |
| Loss Function | CrossEntropyLoss |
| Device | CUDA / CPU |

#### Metrics

| Metric | Value |
|--------|------|
| Accuracy | 0.98 |
| Precision | 0.95 |
| Recall | 0.95 |
| F1-score | 0.95 |
| ROC-AUC | 0.99 |

### Usage

Use the following folder structure:
```
hot-word-detection/
├── data/
│   ├── train/
│   │   ├── stones/         # Audio files with hot word
│   │   ├── not_stones/     # Audio files without hot word
│   │   └── metadata.json   # Dataset metadata
│   ├── extracted/          # Extracted hot word fragments
│   └── test_extract/       # Test outputs
├── models/
│   └── best_model.pth      # Trained model weights
├── notebooks/
│   ├── CNN_detector.ipynb  # Model training notebook
│   ├── data_preparation.ipynb
│   └── EDA.ipynb
├── src/
│   ├── model/
│   │   ├── architecture.py # CNN architecture
│   │   └── model.py        # Model inference
│   ├── services/
│   │   ├── audio_service.py # Audio processing
│   │   └── radio_service.py # Radio streaming
│   ├── utils/
│   │   └── logger.py       # Singleton logger
│   └── constants.py        # Project constants
└── cli.py                  # CLI interface
```

## MLOps Part

It contains code for audio processing, model inference, and radio stream monitoring using CLI.

### Used Tools
- PyTorch
- Librosa
- Soundfile
- numpy
- uv

## How to Run with CLI

### 1. Clone repository and move to project path
```bash
git clone https://github.com/VictoriaKlyueva/hot-word-detection.git
cd hot-word-detection
```

### 2. Install uv
```bash
pip install uv
```

### 3. Install and synchronize project dependencies
```bash
uv sync
```

### Detect hot words in audio file
```bash
python cli.py predict                          # Uses default file (data/thanos_message.wav)
python cli.py predict --file audio.wav         # Custom file
python cli.py predict --output extracted/      # Custom output directory
python cli.py predict -f audio.wav -t 0.7      # Custom file and threshold
```

### Listen to radio stream and detect hot words
```bash
python cli.py radio                            # Default URL, 60 seconds
python cli.py radio --url https://stream.url   # Custom radio URL
python cli.py radio -o output/ -t 0.9          # Custom output and threshold
```

## Model Architecture

CNN-based model for audio classification using MFCC features:
- Input: MFCC + delta + delta-delta (120 features, 101 time frames)
- 3 Convolutional blocks with BatchNorm and MaxPool
- AdaptiveAvgPool + Fully Connected classifier
- Output: 2 classes (stones / not_stones)

## Audio Processing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| SAMPLE_RATE | 16000 | Audio sample rate |
| WINDOW_DURATION | 1.0 | Window size in seconds |
| HOP_DURATION | 0.5 | Hop size in seconds |
| THRESHOLD | 0.88 | Confidence threshold |

## Author

```Klyueva Victoria, 972302```
