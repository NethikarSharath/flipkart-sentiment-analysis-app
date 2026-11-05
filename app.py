
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered"
)


# 2. MODEL CONFIGURATION

MODEL_NAME = "SharathLal/flipkart-distilbert-sentiment-analysis"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3. LOAD TOKENIZER

@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

# 4. LOAD FINE-TUNED MODEL

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return model


# Initialize model and tokenizer
try:
    with st.spinner("Loading the sentiment analysis model..."):
        tokenizer = load_tokenizer()
        model = load_model()

except Exception as e:
    st.error("Unable to load the model. Please check your internet connection.")
    st.exception(e)
    st.stop()


# 5. SENTIMENT PREDICTION FUNCTION

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_class = torch.argmax(probabilities).item()

    sentiment = model.config.id2label[predicted_class]

    confidence = probabilities[predicted_class].item()

    return sentiment, confidence, probabilities


# 6. APPLICATION HEADER

st.title("🛍️ Flipkart Review Sentiment Analyzer")

st.write(
    "Analyze the sentiment of Flipkart product reviews "
    "using a fine-tuned DistilBERT deep learning model."
)

st.caption(
    "Powered by Hugging Face Transformers and PyTorch"
)

st.divider()


# 7. USER INPUT
st.subheader("Enter Your Product Review")

review = st.text_area(
    "Write or paste your review below:",
    placeholder="Example: The product quality is excellent!",
    height=150
)



# 8. SENTIMENT ANALYSIS

if st.button("Analyze Sentiment", type="primary"):

    if review.strip():

        with st.spinner("Analyzing your review..."):

            sentiment, confidence, probabilities = predict_sentiment(
                review.strip()
            )

        st.divider()

        st.subheader("Prediction Result")

        if sentiment == "Positive":
            st.success(f"😊 Predicted Sentiment: {sentiment}")

        elif sentiment == "Negative":
            st.error(f"😞 Predicted Sentiment: {sentiment}")

        else:
            st.info(f"😐 Predicted Sentiment: {sentiment}")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence * 100:.2f}%"
        )

        st.divider()

        # Probability distribution
        st.subheader("Sentiment Probability Distribution")

        for label_id, probability in enumerate(probabilities):

            label = model.config.id2label[label_id]

            probability_value = probability.item()

            st.write(
                f"**{label}: {probability_value * 100:.2f}%**"
            )

            st.progress(probability_value)

    else:

        st.warning("Please enter a product review first.")

# 9. FOOTER


st.divider()

st.info(
    "Predictions are estimates. The model may struggle with neutral, "
    "mixed, or ambiguous reviews."
)

st.caption(
    "Developed as an End-to-End NLP and Deep Learning Project."
)

st.caption(
    "Model: Flipkart-distilbert-sentiment-analysis"
)
