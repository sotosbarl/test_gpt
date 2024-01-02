from transformers import pipeline,AutoModelForSeq2SeqLM
import streamlit as st

model_id = "lmsys/fastchat-t5-3b-v1.0"  # Replace with your model's name on the hub

# pipe = pipeline("text2text-generation", model="lmsys/fastchat-t5-3b-v1.0")

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline("text2text-generation",model=model)

text = st.text_area('enter some text:')
if text:
    out = pipe(text)

    st.json(out)
