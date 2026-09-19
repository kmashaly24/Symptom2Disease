# Symptom2Disease

This project classifies a free-text symptom description into a disease label using a bidirectional LSTM with an attention layer and 100-dimensional GloVe embeddings.

## Project files

- `Symptom2Disease.ipynb`: step-by-step exploratory notebook with explanations, preprocessing, training, evaluation, and attention visualization.
- `symptom2disease.py`: standalone command-line version of the core training and prediction workflow.
- `Symptom2Disease.csv`: labeled symptom descriptions.
- `glove.6B.100d.txt`: GloVe 100-dimensional word vectors.
- `best_model.pt`: saved model weights when available.

## Setup

Use Python 3.10 or newer, create a virtual environment, and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run downloads the NLTK `stopwords` and `wordnet` resources. PyTorch can use a CUDA GPU automatically when a CUDA-enabled installation is available.

## Run the script

Train for the default 100 epochs with early stopping:

```bash
python symptom2disease.py
```

Train for fewer epochs and classify a new symptom description:

```bash
python symptom2disease.py --epochs 20 --predict "fever cough and sore throat"
```

The script reads the CSV and GloVe files from the project directory by default. Use `--data`, `--glove`, and `--checkpoint` to provide different paths. The best checkpoint is saved to `best_model.pt`.

## Run the notebook

Open `Symptom2Disease.ipynb` in VS Code or Jupyter, select a Python kernel, and run the cells from top to bottom. The notebook reads `Symptom2Disease.csv` and `glove.6B.100d.txt` from the project directory.

## Important note

This is an educational machine-learning project, not a medical diagnostic tool. Predictions should not be used to make healthcare decisions.