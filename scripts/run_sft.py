import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Tokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# **データセットの前処理関数**
def preprocess_fn(example):
    if example.get("input"):
        prompt_text = f"User: {example['instruction']}\nAdditional Input: {example['input']}\nAssistant:"
    else:
        prompt_text = f"User: {example['instruction']}\nAssistant:"
    return {
        "prompt": prompt_text,
        "response": example["output"]
    }

def main():
    base_model = "rinna/qwen2.5-bakeneko-32b"
    data_path = "/workspace/data/sft_data.jsonl"
    output_dir = "/workspace/outputs/sft_output"

    # LoRA 設定
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # SFT 設定
    training_args = SFTConfig(
        num_train_epochs=1,
        max_seq_length=512,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        output_dir=output_dir,
        dataset_text_field="prompt",  # ここを "text" ではなく "prompt" にする
        label_names=["labels"]  # `No label_names provided` の警告を防ぐ
    )

    # データセット読み込み
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # トークナイザーの読み込み
    tokenizer = Qwen2Tokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    )

    # LoRA の適用
    model = get_peft_model(model, lora_config)

    # Trainer のセットアップ
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        preprocess_logits_for_metrics=preprocess_fn,  # tokenizer の代わりに `preprocess_fn` を使う
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("SFT training complete.")

if __name__ == "__main__":
    main()
