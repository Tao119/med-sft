# Dockerfile

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# ここで extra-index-url を指定したり、--find-links(-f) を使うなどして
# torch==2.0.1+cu118 をダウンロードできるようにする
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 6. scriptsディレクトリをコピー（任意）
COPY scripts/ /workspace/scripts/

# 7. デフォルトコマンド
CMD ["bash"]
