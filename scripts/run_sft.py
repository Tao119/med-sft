# scripts/run_sft.py

import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model_name = "rinna/qwen2.5-bakeneko-32"
    data_path = "../data/sft_data.jsonl"
    output_dir = "../outputs/sft_output"

    # LoRAなどを用いる設定 (PEFT)
    training_args = SFTConfig(
        num_train_epochs=1,           # デモ用に1epoch
        max_seq_length=512,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        optim="adamw_torch",
        lora=True,                    # LoRAを有効化
        lora_r=8,                     # LoRA rank
        lora_alpha=16,
        lora_dropout=0.05,
        output_dir=output_dir
    )

    # データセット読み込み
    # jsonl形式の場合、"json"オプションで読み込み、split="train"と指定
    dataset = load_dataset("json", data_files=data_path, split="train")

    # "instruction", "input", "output" カラムを使ってSFTする想定
    # ただし、trainer内部では "prompt" と "response" のように命名している可能性があるため、
    # そのマッピング方法をSFTTrainerに渡せるようにする。
    # (trl>=0.7.0以降はSFTTrainerでデフォルト想定カラム: "prompt", "response")
    # → ここでは simplest には下記のように "prompt" と "response" をdatasetに追加して対応する例を示す

    def preprocess_fn(example):
        if example["input"]:
            prompt_text = f"User: {example['instruction']}\nAdditional Input: {example['input']}\nAssistant:"
        else:
            prompt_text = f"User: {example['instruction']}\nAssistant:"
        return {
            "prompt": prompt_text,
            "response": example["output"]
        }

    dataset = dataset.map(preprocess_fn)

    # Tokenizer & Model の読み込み
    # float16/FP16で扱うなら、モデル読み込み時にtorch_dtypeを指定
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Qwen系だとpad_tokenを明示する必要がある場合も
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          # accelerate使用を想定
        torch_dtype=torch.float16,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # 学習後のモデル(LoRAアダプタ)を保存
    trainer.save_model(training_args.output_dir)
    print("SFT training complete and model saved.")


if __name__ == "__main__":
    main()
