# Flipkart Review Sentiment Analyzer

An interactive NLP application that analyzes product reviews and predicts whether their sentiment is **Negative, Neutral, or Positive** using a fine-tuned DistilBERT model.

## Features

- Analyze user-provided product reviews.
- Predict sentiment across three classes.
- Display prediction confidence and class probabilities.
- Load the trained model directly from Hugging Face.
- Provide an interactive interface built with Streamlit.

## Model

- **Architecture:** DistilBERT
- **Base model:** `distilbert/distilbert-base-uncased`
- **Fine-tuned model:** [SharathLal/flipkart-distilbert-sentiment-analysis](https://huggingface.co/SharathLal/flipkart-distilbert-sentiment-analysis)
- **Maximum input length:** 128 tokens

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 93.10% |
| Macro Precision | 80.74% |
| Macro Recall | 75.77% |
| Macro F1-score | 77.60% |

The Neutral class is more difficult to predict because it is underrepresented in the training data.

## Run Locally

### 1. Clone the repository

```bash
git clone [Git Clone Model](https://github.com/NethikarSharath/flipkart-sentiment-analysis-app?utm_source=chatgpt.com)
cd FlipkartSentimentAnalysisApp
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

The application will open at `http://localhost:8501`.

## Limitations

- Predictions are estimates, not guarantees.
- Neutral, mixed, ambiguous, or noisy reviews may be misclassified.
- Confidence scores are not guarantees of prediction correctness.
- The model was trained on Flipkart review data and may not generalize equally well to other domains.

## Technologies

Python · PyTorch · Hugging Face Transformers · Streamlit

## Author

Sharath Nethikar
