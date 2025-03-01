import torch
from datasets import load_dataset
from transformers import Qwen2Tokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from trl import RewardTrainer, RewardConfig

def preprocess_fn(example):
    if "prompt" not in example or "response_1" not in example or "response_2" not in example or "label" not in example:
        raise ValueError(f"Missing necessary keys in example: {example}")

    chosen = example["response_1"] if example["label"] == 1 else example["response_2"]
    rejected = example["response_2"] if example["label"] == 1 else example["response_1"]

    if not chosen or not rejected:
        raise ValueError(f"Invalid chosen/rejected values in example: {example}")

    return {
        "prompt": example["prompt"],
        "chosen": chosen,
        "rejected": rejected
    }

def main():
    base_model = "rinna/qwen2.5-bakeneko-32b"
    data_path = "/workspace/data/reward_data.jsonl"
    output_dir = "/workspace/outputs/reward_output"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    training_args = RewardConfig(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        output_dir=output_dir
    )

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)

    print("Sample data: ", dataset[0])

    try:
        tokenizer = Qwen2Tokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer is None:
            raise ValueError("Tokenizer failed to load. Check the model path.")
    except Exception as e:
        raise RuntimeError(f"Error loading tokenizer: {e}")

    print("Tokenizer loaded successfully:")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        torch_dtype=torch.float16,
    )
    model = get_peft_model(model, lora_config)
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("Reward model training complete.")

if __name__ == "__main__":
    main()
