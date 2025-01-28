from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer from the local directory
model_name_or_path = "./GPT2/filetransfer"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Function to generate text with reduced repetition
def generate_text(prompt, max_length=100, temperature=0.3, top_k=5, top_p=0.9, repetition_penalty=1.2):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=repetition_penalty,
        num_return_sequences=1
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example usage
prompt = "What is a Sales Invoice?"
generated_text = generate_text(prompt)
print(generated_text)