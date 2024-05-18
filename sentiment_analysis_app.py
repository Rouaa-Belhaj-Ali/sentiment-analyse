
import streamlit as st
import psutil
from transformers import AutoTokenizer, AutoModel
import torch

def check_disk_space(min_free_space_gb=0.2):
    return psutil.disk_usage('/').free / (1024 ** 3) >= min_free_space_gb

if not check_disk_space():
    st.error("Not enough disk space. Please free up some space and try again.")
else:
    tokenizer = AutoTokenizer.from_pretrained('Snowflake/snowflake-arctic-embed-xs', trust_remote_code=True)
    model = AutoModel.from_pretrained('Snowflake/snowflake-arctic-embed-xs', trust_remote_code=True, add_pooling_layer=False, safe_serialization=True)
    model.eval()

    def get_text_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            return model(**inputs).last_hidden_state[:, 0, :]

    def analyze_sentiment(embedding):
        return "Positive" if torch.sum(embedding).item() > 0 else "Negative"

    st.title("Streamlit Sentiment Analyzer")
    st.write("Analyze the sentiment of your text using Snowflake's Arctic model family.")
    
    user_input = st.text_area("Enter your text here:")
    if st.button("Analyze Sentiment") and user_input:
        embedding = get_text_embedding(user_input)
        sentiment = analyze_sentiment(embedding)
        st.write(f"The sentiment of the text is: **{sentiment}**")
    elif st.button("Analyze Sentiment"):
        st.write("Please enter some text to analyze.")
