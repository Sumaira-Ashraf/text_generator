import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Initial prompt
input_text = "Once upon a time"

# Generate text iteratively
for i in range(5):  # Adjust the number of iterations as needed
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Set max_new_tokens instead of max_length
    output = model.generate(input_ids, max_new_tokens=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True) 
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    
    # Only keep the newly generated text for the next iteration to avoid exceeding max length
    input_text = generated_text # Update with the recent generated text only
