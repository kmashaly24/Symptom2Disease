"""Train and use a BiLSTM-attention symptom-to-disease classifier."""

from __future__ import annotations

import argparse
import re
import random
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "Symptom2Disease.csv"
DEFAULT_GLOVE = ROOT / "glove.6B.100d.txt"
DEFAULT_CHECKPOINT = ROOT / "best_model.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_nltk() -> tuple[set[str], WordNetLemmatizer]:
    for package in ("stopwords", "wordnet"):
        try:
            nltk.download(package, quiet=True)
        except Exception as error:
            raise RuntimeError(f"Could not download the NLTK resource: {package}") from error
    return set(stopwords.words("english")), WordNetLemmatizer()


def preprocess_text(text: str, english_stopwords: set[str], lemmatizer: WordNetLemmatizer) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in english_stopwords
    )


class Vocabulary:
    def __init__(self) -> None:
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}

    def build(self, sentences: np.ndarray) -> None:
        for sentence in sentences:
            for word in sentence.split():
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)

    def numericalize(self, text: str) -> list[int]:
        unknown = self.word2idx[self.unk_token]
        return [self.word2idx.get(word, unknown) for word in text.split()]

    def __len__(self) -> int:
        return len(self.word2idx)


def load_glove_embeddings(path: Path, vocabulary: Vocabulary, embedding_dim: int) -> torch.Tensor:
    embeddings: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as glove_file:
        for line in glove_file:
            values = line.rstrip().split(" ")
            if len(values) != embedding_dim + 1:
                continue
            embeddings[values[0]] = np.asarray(values[1:], dtype="float32")

    weights = np.zeros((len(vocabulary), embedding_dim), dtype="float32")
    for word, index in vocabulary.word2idx.items():
        if word != vocabulary.pad_token:
            weights[index] = embeddings.get(word, np.random.normal(0, 0.6, embedding_dim))
    return torch.tensor(weights, dtype=torch.float32)


class SymptomDataset(Dataset):
    def __init__(self, texts: np.ndarray, labels: np.ndarray, vocabulary: Vocabulary, max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.vocabulary.numericalize(self.texts[index])[: self.max_len]
        tokens += [self.vocabulary.word2idx[self.vocabulary.pad_token]] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


class Attention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(torch.tanh(self.projection(outputs))).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, embeddings: torch.Tensor) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.lstm(self.embedding(inputs))
        context, weights = self.attention(outputs)
        return self.classifier(context), weights


def metrics(labels: list[int], logits: list[np.ndarray]) -> tuple[float, float, float]:
    predictions = np.argmax(np.asarray(logits), axis=1)
    return (
        f1_score(labels, predictions, average="weighted", zero_division=0),
        precision_score(labels, predictions, average="weighted", zero_division=0),
        recall_score(labels, predictions, average="weighted", zero_division=0),
    )


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    logits_list: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for text, label in loader:
            text, label = text.to(device), label.to(device)
            if training:
                optimizer.zero_grad()
            logits, _ = model(text)
            loss = criterion(logits, label)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            labels.extend(label.cpu().tolist())
            logits_list.extend(logits.detach().cpu().numpy())
    f1, precision, recall = metrics(labels, logits_list)
    return total_loss / len(loader), f1, precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--glove", type=Path, default=DEFAULT_GLOVE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", type=str, help="Predict a disease for one symptom description after training.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    english_stopwords, lemmatizer = prepare_nltk()

    dataframe = pd.read_csv(args.data)
    dataframe["cleaned_text"] = dataframe["text"].astype(str).apply(
        lambda text: preprocess_text(text, english_stopwords, lemmatizer)
    )
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(dataframe["label"])
    texts = dataframe["cleaned_text"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )

    vocabulary = Vocabulary()
    vocabulary.build(x_train)
    max_len = max(1, max(len(text.split()) for text in texts))
    embeddings = load_glove_embeddings(args.glove, vocabulary, 100)
    train_loader = DataLoader(SymptomDataset(x_train, y_train, vocabulary, max_len), batch_size=32, shuffle=True)
    test_loader = DataLoader(SymptomDataset(x_test, y_test, vocabulary, max_len), batch_size=32)
    model = BiLSTMAttentionClassifier(len(vocabulary), 100, 128, len(label_encoder.classes_), embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1 = -1.0
    stale_epochs = 0

    for epoch in range(args.epochs):
        train_values = run_epoch(model, train_loader, criterion, device, optimizer)
        test_values = run_epoch(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss {train_values[0]:.3f} | test F1 {test_values[1]:.3f}")
        if test_values[1] > best_f1:
            best_f1 = test_values[1]
            stale_epochs = 0
            torch.save(model.state_dict(), args.checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for text, label in test_loader:
            logits, _ = model(text.to(device))
            all_labels.extend(label.tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
    print("\nClassification report:\n")
    print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_, zero_division=0))

    if args.predict:
        cleaned = preprocess_text(args.predict, english_stopwords, lemmatizer)
        sample = SymptomDataset(np.array([cleaned]), np.array([0]), vocabulary, max_len)[0][0].unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(sample)[0].argmax(dim=1).item()
        print(f"Predicted disease: {label_encoder.inverse_transform([prediction])[0]}")


if __name__ == "__main__":
    main()