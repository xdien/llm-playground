import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import time
import uvicorn
from fastapi.responses import JSONResponse, StreamingResponse

# Set environment variables to reduce memory fragmentation and optimize CPU usage
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "8"  # Use 8 CPU threads to reduce single-thread load
os.environ["MKL_NUM_THREADS"] = "8"  # Optimize math library

# Config
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B"
max_memory = {
    0: "16GiB",
    1: "16GiB",
    2: "16GiB",
    "cpu": "40GiB"
}

# Manual device map
device_map = {
    "model.embed_tokens": "cpu",  # Offload embedding (~1.56GB)
    "model.norm": "cpu",         # Offload layer norm (~100MB)
    "lm_head": "cpu",            # Offload lm_head to CPU (~1.56GB)
}
# Offload the first 20 layers to CPU, distribute the remaining layers to 3 GPUs with 16GB each (cuda:0, cuda:1, cuda:2)
for i in range(64):  # 64 layers from config.json
    if i < 20:
        device_map[f"model.layers.{i}"] = "cpu"  # First 20 layers offloaded to CPU
    elif i < 34:
        device_map[f"model.layers.{i}"] = "cuda:0"  # 14 layers on GPU 0 (16GB)
    elif i < 48:
        device_map[f"model.layers.{i}"] = "cuda:1"  # 14 layers on GPU 1 (16GB)
    else:
        device_map[f"model.layers.{i}"] = "cuda:2"  # 16 layers on GPU 2 (16GB)

# Ensure offload folder exists
os.makedirs("offload", exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
with torch.no_grad():  # Reduce VRAM usage during loading
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map=device_map,
        offload_folder="offload",
        offload_state_dict=True,
        torch_dtype=torch.float16,
        max_memory=max_memory,
        low_cpu_mem_usage=True
    )

app = FastAPI(title="Qwen3-32B API (transformers)", description="OpenAI-compatible API for Qwen3-32B using transformers")

# Pydantic model for OpenAI-compatible request
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    stream: bool = False
    max_tokens: int = 128  # Reduce max_tokens for faster response
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
def generate_stream(prompt, max_tokens, temperature, top_p, repetition_penalty, num_beams):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda:0")  # Set input_ids on GPU 0
    output_ids = input_ids
    past_key_values = None
    for i in range(max_tokens):
        with torch.no_grad():  # Keep for VRAM saving
            outputs = model(
                input_ids=output_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        logits = outputs.logits[:, -1, :].to("cuda:0")  # Ensure logits on GPU 0
        past_key_values = outputs.past_key_values
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)
        token = tokenizer.decode(next_token_id[0])
        if token.strip() in ["<|endoftext|>", "<|im_end|>"]:
            break
        yield token

# OpenAI-compatible endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req = await request.json()
    messages = req.get("messages", [])
    stream = req.get("stream", False)
    max_tokens = req.get("max_tokens", 128)  # Reduce max_tokens
    temperature = req.get("temperature", 0.7)
    top_p = req.get("top_p", 0.9)
    repetition_penalty = req.get("repetition_penalty", 1.1)
    n = req.get("n", 1)
    model_name = req.get("model", MODEL_NAME)
    num_beams = req.get("num_beams", 1)

    prompt = build_prompt(messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda:0")  # Set input_ids on GPU 0
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=num_beams
    )

    if stream:
        def event_stream():
            generated = ""
            for token in generate_stream(prompt, max_tokens, temperature, top_p, repetition_penalty, num_beams):
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
                yield f"data: {json.dumps(data)}\n\n"
            # End of stream
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        with torch.no_grad():  # Keep for VRAM saving
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
