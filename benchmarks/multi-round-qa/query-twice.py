import sys
import time
from openai import OpenAI
from transformers import AutoTokenizer

client = OpenAI(
    api_key="dummy-key",  # required by OpenAI client even for local servers
    base_url="http://localhost:8000/v1"
)

models = client.models.list()
model = models.data[0].id

# 119512 characters total
# 26054 tokens total
long_context = ""
with open("man-bash.txt", "r") as f:
    long_context = f.read()

# a truncation of the long context for the --max-model-len 16384
# if you increase the --max-model-len, you can decrease the truncation i.e.
# use more of the long context
long_contexts = [
    long_context[1300:71300],
    long_context[1400:71400],
    long_context[1500:71500],
    long_context[1600:71600],
]

tokenizer = AutoTokenizer.from_pretrained("/models/preset/meta-llama/Meta-Llama-3.1-70B-Instruct/v1.0/")
question = "Summarize bash in 2 sentences."

def query_and_measure_ttft(prompt):
    start = time.perf_counter()
    ttft = None

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.7,
        stream=True,
    )

    for chunk in chat_completion:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            if ttft is None:
                ttft = time.perf_counter()
            print(chunk_message, end="", flush=True, file=sys.stderr)

    print("\n", file=sys.stderr)  # New line after streaming
    return ttft - start

cold_ttfts = []
for i in range(4):
    prompt = f"{long_contexts[i]}\n\n{question}"
    print(f"Number of tokens in prompt {i}: {len(tokenizer.encode(prompt))}")

    print(f"Querying vLLM server with cold LMCache CPU Offload with prompt {i}")
    cold_ttft = query_and_measure_ttft(prompt)
    print(f"Cold TTFT: {cold_ttft:.3f} seconds")
    cold_ttfts.append(cold_ttft)

for i in range(4):
    prompt = f"{long_contexts[i]}\n\n{question}"
    print(f"Number of tokens in prompt {i}: {len(tokenizer.encode(prompt))}")

    print(f"\nQuerying vLLM server with warm LMCache CPU Offload with prompt {i}")
    warm_ttft = query_and_measure_ttft(prompt)
    print(f"Warm TTFT: {warm_ttft:.3f} seconds")

    print(f"\nTTFT Improvement: {(cold_ttfts[i] - warm_ttft):.3f} seconds \
    ({(cold_ttfts[i]/warm_ttft):.1f}x faster)")