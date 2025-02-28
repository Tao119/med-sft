# scripts/test_sft.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    name = input("name?")
    model_path =f"../outputs/{name}_output" if name != "base" else "rinna/qwen2.5-bakeneko-32b"
    print(f"using model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,   # カスタムモデル実装を読み込む
        token=True,
        device_map="auto",        # GPUを自動割り当て
        torch_dtype=torch.float16 # 半精度指定（任意）
    )

    # 推論用のプロンプト例
    prompt = "風邪を引いたときの早めの回復方法を教えてください。"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Prompt:", prompt)
    print("Generated:", generated_text)

if __name__ == "__main__":
    main()
