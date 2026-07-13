import os, subprocess, sys
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle, numpy as np

MODEL="one_to_many_rnn.keras"
TOKENIZER="tokenizer.pkl"
DATASET="technology_sentences.csv"

if not os.path.exists(MODEL) or not os.path.exists(TOKENIZER):
    subprocess.run([sys.executable,"train.py"],check=True)

with open(TOKENIZER,"rb") as f:
    tok=pickle.load(f)
model=load_model(MODEL,compile=False)

max_input_len=3

def generate(keyword):
    seq=tok.texts_to_sequences([keyword.lower()])
    seq=pad_sequences(seq,maxlen=max_input_len,padding="post")
    pred=model.predict(seq,verbose=0)
    idx=np.argmax(pred,axis=-1)[0]
    rev={v:k for k,v in tok.word_index.items()}
    return " ".join(rev.get(i,"") for i in idx if i!=0)

st.title("One-to-Many RNN")
k=st.text_input("Keyword")
if st.button("Generate"):
    st.success(generate(k))
