import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "FenomAI/Qwen3-VL-4B-Instruct-AWQ-4bit"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("MODEL LOADED SUCCESSFULLY")
print("GPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

image_path = input("\nEnter image path: ").strip().strip('"')

question = input(
    "Question [default: What's wrong with this object?]: "
).strip()

if not question:
    question = "What's wrong with this object?"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": question,
            },
        ],
    }
]

print("\nPreparing image...")

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Put tensors on the model's execution device.
inputs = {
    k: v.to(model.device) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

print("Running inference...\n")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
       do_sample=False,
       use_cache=True,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(
        inputs["input_ids"],
        generated_ids
    )
]

output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("=" * 60)
print("QWEN3-VL-4B RESPONSE")
print("=" * 60)
print(output[0])
print("=" * 60)