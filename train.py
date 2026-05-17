import argparse
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertModel


def parse_args():
    parser = argparse.ArgumentParser(description="BERT + ResNet50 Multimodal Emotion Detection")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing training video files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--max_seq_len", type=int, default=100)
    return parser.parse_args()


def get_video_path(row, video_dir):
    filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
    return os.path.join(video_dir, filename)


def extract_bert_embeddings(text_list, tokenizer, bert_model, max_length):
    embeddings = []
    for text in tqdm(text_list, desc="BERT embeddings"):
        inputs = tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.vstack(embeddings)


def extract_visual_features(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (224, 224)))
    cap.release()
    return np.mean(frames, axis=0) if frames else np.zeros((224, 224, 3))


def build_model(num_classes):
    # Text branch
    input_text = Input(shape=(768,), name="text_input")
    x_text = Dense(128, activation="relu")(input_text)
    x_text = Dropout(0.5)(x_text)

    # Visual branch
    input_visual = Input(shape=(224, 224, 3), name="visual_input")
    resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x_visual = resnet(input_visual)
    x_visual = GlobalAveragePooling2D()(x_visual)
    x_visual = Dense(128, activation="relu")(x_visual)

    # Fusion
    merged = Concatenate()([x_text, x_visual])
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=[input_text, input_visual], outputs=output)
    return model


def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


def main():
    args = parse_args()

    df = pd.read_csv(args.train_csv, encoding="ISO-8859-1")
    df["video_path"] = df.apply(lambda r: get_video_path(r, args.video_dir), axis=1)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Sentiment"])
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {list(label_encoder.classes_)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()

    X_text = extract_bert_embeddings(
        df["Utterance"].astype(str).tolist(), tokenizer, bert_model, args.max_seq_len
    )

    X_visual = np.array(
        [extract_visual_features(p) for p in tqdm(df["video_path"], desc="Video frames")]
    ) / 255.0

    y = to_categorical(df["label"], num_classes=num_classes)

    X_text_train, X_text_val, X_vis_train, X_vis_val, y_train, y_val = train_test_split(
        X_text, X_visual, y, test_size=args.val_split, random_state=42
    )

    model = build_model(num_classes)
    model.compile(
        optimizer=Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit(
        [X_text_train, X_vis_train], y_train,
        validation_data=([X_text_val, X_vis_val], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    loss, accuracy = model.evaluate([X_text_val, X_vis_val], y_val)
    print(f"\nValidation Loss:     {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    model.save("model.h5")
    print("Model saved to model.h5")

    plot_history(history)


if __name__ == "__main__":
    main()
