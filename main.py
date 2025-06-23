import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import os

# Model config
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto"
)

app = FastAPI(title="Qwen2.5-7B API (transformers)", description="OpenAI-compatible API for Qwen2.5-7B using transformers")

# Pydantic model for OpenAI-compatible request
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    stream: bool = False
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    n: int = 1

# Helper to build prompt from messages
def build_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|user|> {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|> {msg['content']}\n"
    prompt += "<|assistant|> "
    return prompt

# Streaming generator
def generate_stream(prompt, max_tokens, temperature, top_p, repetition_penalty):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    output_ids = input_ids
    past_key_values = None
    for i in range(max_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=output_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)
            token = tokenizer.decode(next_token_id[0])
            if token.strip() == "<|endoftext|>" or token.strip() == "<|im_end|>":
                break
            yield token

# OpenAI-compatible endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req = await request.json()
    messages = req.get("messages", [])
    stream = req.get("stream", False)
    max_tokens = req.get("max_tokens", 2048)
    temperature = req.get("temperature", 0.7)
    top_p = req.get("top_p", 0.9)
    repetition_penalty = req.get("repetition_penalty", 1.1)
    n = req.get("n", 1)
    model_name = req.get("model", MODEL_NAME)

    prompt = build_prompt(messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    if stream:
        def event_stream():
            generated = ""
            for token in generate_stream(prompt, max_tokens, temperature, top_p, repetition_penalty):
                generated += token
                data = {
                    "id": f"chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "delta": {"content": token},
                            "index": 0,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {data}\n\n"
            # End of stream
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                **gen_kwargs
            )
        output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_ids.shape[-1],
                "completion_tokens": len(tokenizer.encode(output_text)),
                "total_tokens": input_ids.shape[-1] + len(tokenizer.encode(output_text))
            }
        }
        return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 