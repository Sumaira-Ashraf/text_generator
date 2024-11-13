import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text(prompt, max_length=50, num_beams=5, no_repeat_ngram_size=2):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("Text Generator")

prompt = st.text_input("Enter your prompt:")
max_length = st.slider("Max Length", 10, 100, 50)
num_beams = st.slider("Number of Beams", 1, 5, 2)
no_repeat_ngram_size = st.slider("No Repeat N-Gram Size", 1, 5, 2)

if st.button("Generate"):
    generated_text = generate_text(prompt, max_length, num_beams, no_repeat_ngram_size)
    st.write(generated_text)




