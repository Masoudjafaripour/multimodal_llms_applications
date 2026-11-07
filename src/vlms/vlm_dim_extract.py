# ====================================================
# 📦 IMPORTS
# ====================================================
import os, cv2, json, base64, torch
from transformers import (
    AutoProcessor, AutoModelForVision2Seq,
    AutoModelForCausalLM, AutoTokenizer,
    DonutProcessor, VisionEncoderDecoderModel,
    LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering,
    BitsAndBytesConfig
)
from openai import OpenAI
from PIL import Image

# ====================================================
# 🔧 CONFIG
# ====================================================
openai_api_key = "YOUR_OPENAI_KEY"

MODEL_QWEN = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_LAYOUTLM = "microsoft/layoutlmv3-base"
MODEL_DONUT = "naver-clova-ix/donut-base-finetuned-docvqa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "results/blueprints/floorplan3.jpg"   # your input blueprint
MAX_NEW_TOKENS = 256


# ====================================================
# 🧩 HELPER: encode image as base64
# ====================================================
def encode_image_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ====================================================
# 1️⃣  GPT-4o  (API-based)
# ====================================================
def extract_with_gpt4o(image_path):
    client = OpenAI(api_key=openai_api_key)
    frame_b64 = encode_image_base64(image_path)

    prompt_text = """You are analyzing an architectural blueprint.
    Extract all measurements, materials, and room names visible in the image.
    Return structured JSON in this format:
    {
      "rooms": [{"name": "...", "dimensions": "..."}],
      "materials": [{"element": "...", "type": "..."}]
    }
    ⚠️ Only return valid JSON, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user",
             "content": [
                 {"type": "text", "text": prompt_text},
                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
             ]}
        ],
        max_tokens=500
    )

    reply = response.choices[0].message.content.strip()
    print("\n[GPT-4o] raw:", reply)
    try:
        return json.loads(reply)
    except Exception:
        return {"raw": reply}


# ====================================================
# 2️⃣  Qwen2.5-VL (local open-source)
# ====================================================
def extract_with_qwen(image_path):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(MODEL_QWEN, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_QWEN,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()

    img = Image.open(image_path).convert("RGB")
    query = (
        "Extract all numeric dimensions and material annotations "
        "from this architectural drawing. "
        "Return strictly valid JSON with keys 'dimensions' and 'materials'."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image", "image": img},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[img], return_tensors="pt").to(DEVICE)

    generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    out = processor.batch_decode(generated, skip_special_tokens=True)[0]

    print("\n[Qwen2.5-VL] raw:", out)
    try:
        return json.loads(out[out.find("{"):out.rfind("}")+1])
    except Exception:
        return {"raw": out}



# ====================================================
# 3️⃣  LayoutLMv3  (document-style VLM)
# ====================================================
def extract_with_layoutlm(image_path, question="What are the room dimensions?"):
    processor = LayoutLMv3Processor.from_pretrained(MODEL_LAYOUTLM)
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(MODEL_LAYOUTLM).to(DEVICE)

    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, question, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    start, end = outputs.start_logits.argmax(), outputs.end_logits.argmax()
    answer = processor.tokenizer.decode(inputs["input_ids"][0][start:end+1])
    print("\n[LayoutLMv3] answer:", answer)
    return {"text_answer": answer}


# ====================================================
# 4️⃣  Donut (OCR-free doc-VQA)
# ====================================================
def extract_with_donut(image_path):
    processor = DonutProcessor.from_pretrained(MODEL_DONUT)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DONUT).to(DEVICE)
    img = Image.open(image_path).convert("RGB")
    task_prompt = "<s_docvqa><s_question>List all room dimensions visible in the blueprint.</s_question><s_answer>"
    inputs = processor(img, task_prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    print("\n[Donut] raw:", decoded)
    return {"text_answer": decoded}


# ====================================================
# 🚀 RUN ALL
# ====================================================
if __name__ == "__main__":
    print("Running blueprint extraction benchmark...\n")

    res_gpt4o = extract_with_gpt4o(IMAGE_PATH)
    res_qwen = extract_with_qwen(IMAGE_PATH)
    res_layoutlm = extract_with_layoutlm(IMAGE_PATH)
    res_donut = extract_with_donut(IMAGE_PATH)

    results = {
        "gpt4o": res_gpt4o,
        "qwen2.5vl": res_qwen,
        "layoutlmv3": res_layoutlm,
        "donut": res_donut
    }

    print("\n=== Final Parsed Outputs ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
