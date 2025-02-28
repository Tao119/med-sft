from huggingface_hub import create_repo, HfApi, upload_folder
name = input("name")
type = input("type")

REPO_ID = "Tao119/test-{name}"

# リポジトリ作成（既に作成済みならskip）
api = HfApi()
api.create_repo(repo_id=REPO_ID, exist_ok=True)

# フォルダをアップロード
upload_folder(
    folder_path="../outputs/{type}_output",
    repo_id=REPO_ID,
    repo_type="model"
)
print(f"Model uploaded to https://huggingface.co/{REPO_ID}")
