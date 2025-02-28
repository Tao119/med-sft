import torch
from datasets import load_dataset
from transformers import Qwen2Tokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from trl import RewardTrainer, RewardConfig

def preprocess_fn(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["response_1"] if example["label"] == 1 else example["response_2"], 
        "rejected": example["response_2"] if example["label"] == 1 else example["response_1"], 
    }

def main():
    base_model = "rinna/qwen2.5-bakeneko-32b"
    data_path = "/workspace/data/reward_data.jsonl"
    output_dir = "/workspace/outputs/reward_output"

    # Reward モデルの設定
    training_args = RewardConfig(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        output_dir=output_dir
    )

    # データセットの読み込み
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(preprocess_fn)
    print("Dataset loaded successfully. Features:", dataset.features)

    # トークナイザーの読み込み
    tokenizer = Qwen2Tokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )

    # **トークナイザーが適切にロードされたか確認**
    if tokenizer is None:
        raise ValueError("Tokenizer failed to load. Check the model path.")

    print("Tokenizer loaded successfully:", tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # データのコラレータ（バッチ処理用）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # モデルの読み込み（分類用）
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,  # 分類タスクの出力を1つに設定
        torch_dtype=torch.float16,
    )

    # Trainer のセットアップ
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator  # `tokenizer` を削除し `data_collator` のみ渡す
    )

    # トレーニングの実行
    trainer.train()
    trainer.save_model(output_dir)
    print("Reward model training complete.")

if __name__ == "__main__":
    main()
