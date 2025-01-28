from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained GPT-2 model and tokenizer
model_name_or_path = "./GPT2/filetransfer"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Define the request body structure
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2

# Function to ensure the text ends at a natural stopping point
def post_process_text(text):
    # Find the last occurrence of a sentence-ending punctuation
    end_punctuation = [".", "!", "?"]
    for punct in end_punctuation:
        if punct in text:
            text = text[:text.rfind(punct) + 1]
            break
    return text


# Define the text generation endpoint
@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    try:
        inputs = tokenizer.encode(request.prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=True,
            repetition_penalty=request.repetition_penalty,
            num_return_sequences=1
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = post_process_text(text)
        return {"generated_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)