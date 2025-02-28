import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

def main():
    base_model_path = "/workspace/outputs/sft_output"
    data_path = "/workspace/data/dpo_data.jsonl"
    output_dir = "/workspace/outputs/dpo_output"

    # DPO の設定
    dpo_config = DPOConfig(
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        output_dir=output_dir
    )

    # データセットの読み込み
    dataset = load_dataset("json", data_files=data_path, split="train")
    print("Dataset loaded successfully. Features:", dataset.features)  # デバッグ用

    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # DPOTrainer のセットアップ
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=dataset
    )

    # トレーニングの実行
    trainer.train()
    trainer.save_model(output_dir)
    print("DPO training complete.")

if __name__ == "__main__":
    main()
