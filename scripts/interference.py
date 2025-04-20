import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

# Load base model and LoRA adapter
base_model = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(base_model)
tokenizer = LlamaTokenizer.from_pretrained(base_model)

model = PeftModel.from_pretrained(model, "models/llama-lora-checkpoints")
model.eval()


def generate_response(instruction, context=None, max_tokens=200):
    if context:
        prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}"
    else:
        prompt = f"### Instruction:\n{instruction}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example
response = generate_response(
    instruction="How does photosynthesis work?",
    context="In biology, photosynthesis is a process used by plants to convert light energy into chemical energy."
)
print("\n🧠 Generated Response:\n", response)
