from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

model_name = "TheBloke/dolphin-2.6-mistral-7B-dpo-laser-GPTQ"


prompt = "what is AI ?"


tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, enforce_eager=True, dtype=torch.float16)

sampling_params = SamplingParams(
    n=1,
    best_of=5,
    use_beam_search=True,
    stop_token_ids=[tokenizer.eos_token_id],
    max_tokens=300,
    temperature=0,
)


outputs = llm.generate(prompt, sampling_params)[0]
generated_query = outputs.outputs[0].text

print("====================================")
print(generated_query)
print("====================================")
