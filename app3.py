import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

#setting page config
st.set_page_config(page_title='Financial Table Classifier',layout="wide",initial_sidebar_state="expanded")
st.header(':green[Financial Table Classifier]',divider = 'rainbow')

# Function to extract tables from an HTML file
def extract_tables_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = []
    for table in soup.find_all('table'):
        df = pd.read_html(str(table))[0]
        tables.append(df)
    return tables

# Convert tables to text
def tables_to_text(tables):
    text_data = []
    for table in tables:
        text = ' '.join(table.astype(str).values.flatten())
        text_data.append(text)
    return text_data

# Load the trained model and vectorizer
model = joblib.load('financial_table_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

image_path = 'C:/Users/manik/Desktop/finacplus/OIP.jpeg' 

# Embedding inline CSS for background color and image
st.markdown(
    f"""
    <style>
    body {{
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), url('{image_path}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main {{
        background: rgba(255, 255, 255, 0.8);  /* White background with opacity */
        padding: 20px;
        border-radius: 10px;
    }}
    header, .stApp {{
        background: rgba(255, 255, 255, 0.8);  /* White background with opacity */
    }}
    .stButton>button {{
        color: #fff;
        background-color: #007BFF;
        border-color: #007BFF;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }}
    .stTitle, .stMarkdown {{
        text-align: center;
        color: #333;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Streamlit app title and description
st.markdown("### :blue[Upload an HTML file containing financial tables, and the model will classify each table.]")

# File uploader for HTML file
uploaded_file = st.file_uploader("Choose an HTML file", type="html")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.html", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract tables from the uploaded file
    tables = extract_tables_from_html("temp.html")
    if tables:
        st.write(f"Extracted {len(tables)} tables from the uploaded file.")
        
        # Convert tables to text
        text_data = tables_to_text(tables)
        
        # Transform text data using the vectorizer
        X = vectorizer.transform(text_data)
        
        # Predict the category for each table
        predictions = model.predict(X)
        
        # Display the results
        for i, (table, prediction) in enumerate(zip(tables, predictions)):
            st.write(f"### Table {i+1} - Predicted Category: {prediction}")
            st.write(table)
    else:
        st.write("No tables found in the uploaded HTML file.")
    
    # Clean up the temporary file
    os.remove("temp.html")


