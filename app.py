import streamlit as st
from nlp_models import analyze_text
from image_model import classify_image

st.title("Smart Support Assistant")

# Text input
text_input = st.text_area("Enter support query:")
if text_input:
    result = analyze_text(text_input)
    st.write("Entities:", result["entities"])
    st.write("Sentiment:", result["sentiment"])

# Image input
image_file = st.file_uploader("Upload product image", type=["jpg", "png"])
if image_file:
    with open("temp_image.jpg", "wb") as f:
        f.write(image_file.read())
    label = classify_image("temp_image.jpg")
    st.write(f"Predicted label: {label}")
