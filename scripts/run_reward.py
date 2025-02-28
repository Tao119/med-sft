# scripts/run_reward.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig


def main():
    model_name = "rinna/qwen2.5-bakeneko-32"
    data_path = "../data/reward_data.jsonl"
    output_dir = "../outputs/reward_output"

    # Rewardモデル用の設定
    training_args = RewardConfig(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        output_dir=output_dir
    )

    # データ読み込み
    # カラム: "prompt", "response_1", "response_2", "label" (1 or 2)
    dataset = load_dataset("json", data_files=data_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Qwen系モデルでpad_token_idが存在しないなら設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # RewardモデルとしてSequenceClassificationを用いる
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Rewardは1次元のスコアを出す
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Trainerを作成
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("Reward model training complete.")


if __name__ == "__main__":
    main()
