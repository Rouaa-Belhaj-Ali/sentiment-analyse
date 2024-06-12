Streamlit Sentiment Analyzer
This repository contains a Streamlit application that analyzes the sentiment of a given text using the Snowflake Arctic model family. The application leverages the transformers library by Hugging Face and psutil for disk space management.

###Features
Disk Space Check: Ensures there is enough free disk space before running the sentiment analysis.
Text Embedding: Uses the AutoTokenizer and AutoModel from the Snowflake Arctic model family to convert input text into embeddings.
Sentiment Analysis: Determines if the sentiment of the text is positive or negative based on the embeddings.
Streamlit Interface: Provides a simple and interactive web interface for users to input text and view sentiment results.
Installation
Clone the repository:

git clone https://github.com/Rouaa-Belhaj-Ali/streamlit-sentiment-analyzer.git
cd streamlit-sentiment-analyzer
Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:


pip install -r requirements.txt
Run the application:


streamlit run app.py
Usage
Ensure that your system has at least 0.2 GB of free disk space. The application will not run without sufficient disk space.
Open your web browser and go to the local URL provided by Streamlit (usually http://localhost:8501).
Enter the text you want to analyze in the text area.
Click the "Analyze Sentiment" button.
View the sentiment result displayed below the button.
Code Overview
Disk Space Check: The check_disk_space function checks if there is enough free disk space before proceeding with the sentiment analysis.
python

import psutil

def check_disk_space(min_free_space_gb=0.2):
    return psutil.disk_usage('/').free / (1024 ** 3) >= min_free_space_gb
Model Initialization: Loads the tokenizer and model from the Snowflake Arctic model family.

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('Snowflake/snowflake-arctic-embed-xs', trust_remote_code=True)
model = AutoModel.from_pretrained('Snowflake/snowflake-arctic-embed-xs', trust_remote_code=True, add_pooling_layer=False, safe_serialization=True)
model.eval()
Text Embedding: Converts the input text into embeddings.
python
Copier le code
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[:, 0, :]
Sentiment Analysis: Analyzes the sentiment based on the text embeddings.
python
Copier le code
def analyze_sentiment(embedding):
    return "Positive" if torch.sum(embedding).item() > 0 else "Negative"
Streamlit Interface: Provides the user interface for input and output.
python
Copier le code
import streamlit as st

st.title("Streamlit Sentiment Analyzer")
st.write("Analyze the sentiment of your text using Snowflake's Arctic model family.")

user_input = st.text_area("Enter your text here:")
if st.button("Analyze Sentiment") and user_input:
    embedding = get_text_embedding(user_input)
    sentiment = analyze_sentiment(embedding)
    st.write(f"The sentiment of the text is: **{sentiment}**")
elif st.button("Analyze Sentiment"):
    st.write("Please enter some text to analyze.")
Requirements
streamlit
psutil
transformers
torch
###License
This project is licensed under the MIT License. See the LICENSE file for details.

###Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

