import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

def main():
    policy_model_path = "/workspace/outputs/sft_output"
    reward_model_path = "/workspace/outputs/reward_output"
    data_path = "/workspace/data/ppo_prompts.jsonl"
    output_dir = "/workspace/outputs/ppo_output"

    # PPO の設定
    ppo_config = PPOConfig(
        model_name=policy_model_path,
        learning_rate=1e-6,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=512,
        log_with=None,
        output_dir=output_dir
    )

    # データセットの読み込み
    dataset = load_dataset("json", data_files=data_path, split="train")
    print("Dataset loaded successfully. Features:", dataset.features)  # デバッグ用

    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # モデルの読み込み
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 参照モデル（Reference Model）の作成
    ref_model = create_reference_model(model)

    # 報酬モデル（Reward Model）の読み込み
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # PPOTrainer のセットアップ
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        dataset=dataset
    )

    # 1エポックのみデモ実行（データ数に応じて調整）
    for epoch in range(1):
        for sample in dataset:
            prompt = sample["prompt"]
            gen_tokens = ppo_trainer.generate(prompt, max_new_tokens=50)
            response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            ppo_trainer.step(prompt, response)

    # モデルの保存
    ppo_trainer.save_model(output_dir)
    print("PPO training complete.")

if __name__ == "__main__":
    main()
