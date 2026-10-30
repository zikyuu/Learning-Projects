"""Thin wrapper over Ollama's local HTTP chat API (this is all a 'client library' really is)."""
import requests

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"

def chat(messages, tools=None, temperature=0.0, model=MODEL):
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096}}
    if tools:
        body["tools"] = tools
    r = requests.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]   # {'role','content', optional 'tool_calls'}
