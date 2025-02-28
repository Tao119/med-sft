# scripts/run_dpo.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


def main():
    # SFT済みのモデルをベースにDPOする想定 (LoRAアダプタ入り)
    # ここではSFTの出力 "../outputs/sft_output" を読み込む
    base_model_path = "../outputs/sft_output"
    data_path = "../data/dpo_data.jsonl"
    output_dir = "../outputs/dpo_output"

    dpo_config = DPOConfig(
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        output_dir=output_dir
    )

    dataset = load_dataset("json", data_files=data_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # DPOTrainer初期化
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=dataset,
        # validation_dataset=...,   # バリデーション用があれば指定
    )

    # 学習
    trainer.train()
    trainer.save_model(output_dir)
    print("DPO training complete.")


if __name__ == "__main__":
    main()
