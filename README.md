# SPUMix

Official code for **SPUMix** (*Style-Paired Uncertainty-Aware Mixstyle*), a feature-level style mixing module for multi-view pedestrian detection.

SPUMix is integrated into the ResNet-18 backbone of an [MVDeTr](https://github.com/hou-yz/MVDeTr)-style detector. During training it pairs samples with the most distant feature styles and mixes channel-wise statistics with an uncertainty-aware mixing coefficient, improving robustness under appearance / domain shifts.

Training is recommended on a GPU with memory comparable to or larger than an RTX 4090.

## Dependencies

- Python 3.7+
- PyTorch 1.7.1 & torchvision 0.8.2
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- motmetrics
- MATLAB & matlabengine (for the official MOTChallenge evaluation kit; a Python alternative is available under `multiview_detector/evaluation/pyeval`)

Build the deformable attention ops (forked from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)):

```bash
cd multiview_detector/models/ops
bash make.sh
```

## Data Preparation

Download the datasets from their official sources and place them under `Data/`:

| Dataset | Link |
|---------|------|
| Wildtrack | https://www.epfl.ch/labs/cvlab/data/data-wildtrack/ |
| MultiviewX | https://github.com/hou-yz/MVDet/ |
| MVPerception (Day-to-Night) | https://drive.google.com/file/d/110ZbfWw2oDv0arC1Dzha_O7J-rjzZVvq/view?usp=sharing |

Expected layout:

```
Data/
├── MultiviewX/
├── Wildtrack/
└── MVPerception/
```

## Training

```bash
# Wildtrack
python main.py -d wildtrack --spumix_p 0.5 --spumix_alpha 0.1

# MultiviewX
python main.py -d multiviewx --spumix_p 0.5 --spumix_alpha 0.1

# MVPerception
python main.py -d mvperception --spumix_p 0.5 --spumix_alpha 0.1
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-d`, `--dataset` | `wildtrack` | Dataset: `wildtrack`, `multiviewx`, or `mvperception` |
| `--spumix_p` | `0.5` | SPUMix mixing probability \(P\) (`0` disables SPUMix) |
| `--spumix_alpha` | `0.1` | SPUMix Beta distribution parameter \(\alpha\) |

## Testing & Cross-Dataset Evaluation

After training, `main.py` evaluates the model on the corresponding test split each epoch.

To evaluate MultiviewX-trained checkpoints on Wildtrack (cross-dataset):

```bash

python test_multiviewx_models_on_wildtrack.py

```

## Pre-Trained Models

Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1C56ayW3wWaQG5M7ZGCUNt1jDSlm6H3YH/view?usp=sharing).

