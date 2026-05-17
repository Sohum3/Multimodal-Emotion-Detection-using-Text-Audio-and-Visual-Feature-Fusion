# Multimodal Emotion Detection

Sentiment classification from video utterances using a late-fusion architecture that combines **BERT** (text) and **ResNet50** (video frames).

## Architecture

```
Utterance text ──► BERT (bert-base-uncased) ──► mean pool ──► Dense(128) ──┐
                                                                             ├──► Concat ──► Dense(128) ──► Softmax
Video clip ──────► ResNet50 (ImageNet) ──► GlobalAvgPool ──► Dense(128) ──┘
```

- **Text branch:** BERT mean-pooled token embeddings (768-dim) → Dense(128) + Dropout(0.5)  
- **Visual branch:** All frames averaged per clip → ResNet50 backbone → GlobalAvgPool → Dense(128)  
- **Fusion:** Concatenation → Dense(128) + Dropout(0.5) → Softmax over 3 classes (negative / neutral / positive)

## Dataset

The model was trained on the [MELD dataset](https://affective-meld.github.io/) — multimodal emotion lines from the Friends TV series. Each sample is a video clip of a single utterance paired with its transcript and a sentiment label.

Video files follow the naming convention: `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`

> The raw data is not included in this repo. Download it separately and place it under `data/`.

Expected folder layout:

```
data/
├── train_csv/
│   └── train.csv
├── train_videos/
│   └── dia0_utt0.mp4 ...
├── test_csv/
│   └── test.csv
└── test_videos/
    └── ...
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and a CUDA-capable GPU for reasonable training speed.

## Training

```bash
python train.py \
  --train_csv data/train_csv/train.csv \
  --video_dir data/train_videos \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4
```

After training, `model.h5` and `training_curves.png` are saved to the working directory.

## Results

| Split      | Accuracy |
|------------|----------|
| Validation | ~XX%     |

## Dependencies

- TensorFlow / Keras — model training
- PyTorch + HuggingFace Transformers — BERT embeddings
- OpenCV — video frame extraction
- scikit-learn — label encoding, train/val split
