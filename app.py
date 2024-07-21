import streamlit as st
from prediction import predict_next_word

st.title("Shakespear Hamlet Prediction")

input_text = st.text_input("Enter word sequence:",
                           "We do it wrong, being so Maiesticall")

if st.button("Predict next word"):
    next_word = predict_next_word(input_text)
    st.write(f"Next word: {next_word}")
