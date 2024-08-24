import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

snapshot_download(repo_id="wangrui6/Zhihu-KOL",
                  repo_type="dataset",
                  local_dir="data",
                  max_workers=8,
                  resume_download=True
                  )
snapshot_download(
    repo_id="internlm/internlm2_5-1_8b-chat",
    local_dir="model",
    resume_download=True,
    max_workers=8
)