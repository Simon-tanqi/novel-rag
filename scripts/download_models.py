"""
download_models.py — 首次运行自动下载模型到项目本地 models/ 目录

功能：
  - 自动创建 models/ 目录（.gitignore 忽略它）
  - 下载 BAAI/bge-small-zh-v1.5（≈ 92MB，CPU 向量编码）
  - 下载 BAAI/bge-base-zh-v1.5（≈ 370MB，精度更高）
  - 下载 BAAI/bge-reranker-base（≈ 220MB，重排模型）
  - 若已存在则跳过（按 commit hash 校验）

用法（首次 clone 后运行一次）：
    python scripts/download_models.py

    # 也可单独下载某类模型：
    python scripts/download_models.py --embedding-small   # 仅 bge-small（默认）
    python scripts/download_models.py --embedding-base    # bge-base
    python scripts/download_models.py --reranker          # 重排模型
    python scripts/download_models.py --all               # 全套（默认）

    # 国内用户：
    set HF_ENDPOINT=https://hf-mirror.com
    python scripts/download_models.py --all

输出到：
    models/
      bge-small-zh-v1.5/          ← bge-small 本地目录
      bge-base-zh-v1.5/           ← bge-base 本地目录
      bge-reranker-base/          ← 重排模型本地目录

config.json 写入：
    "embedding_model_path": "models/bge-small-zh-v1.5"
    "reranker_model_path":  "models/bge-reranker-base"
"""

from __future__ import annotations
import os
import sys
import argparse
import hashlib
import json
from pathlib import Path

# HF 下载兼容国内
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT or "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / ".manifest.json"  # 记录已下载 commit hash


def get_model_info():
    return [
        {
            "id": "BAAI/bge-small-zh-v1.5",
            "local_dir": "bge-small-zh-v1.5",
            "size_mb": 92,
            "type": "embedding",
            "description": "轻量嵌入（CPU 可跑，推理快）",
        },
        {
            "id": "BAAI/bge-base-zh-v1.5",
            "local_dir": "bge-base-zh-v1.5",
            "size_mb": 370,
            "type": "embedding",
            "description": "高精度嵌入（推荐 GPU，精度更好）",
        },
        {
            "id": "BAAI/bge-reranker-base",
            "local_dir": "bge-reranker-base",
            "size_mb": 220,
            "type": "reranker",
            "description": "重排模型（CrossEncoder，提升召回精度）",
        },
    ]


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_local_commit(model_dir: Path) -> str | None:
    """读取本地模型的 commit hash（用于校验是否需更新）"""
    ref_file = model_dir / ".gitignore"  # HF 会在顶层放 refs 文件
    refs_dir = model_dir / "refs"
    if refs_dir.exists():
        heads = list(refs_dir.glob("**/*"))
        if heads:
            return heads[0].read_text(encoding="utf-8").strip()
    return None


def download_model(model_id: str, local_dir: Path, *, desc: str, size_mb: int):
    """下载单个模型到本地目录"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_dir_abs = MODELS_DIR / local_dir
    manifest = load_manifest()
    old_commit = manifest.get(model_id, {}).get("commit")

    if local_dir_abs.exists():
        # 已存在，跳过（精确校验暂用大小粗估）
        mb = sum(
            f.stat().st_size for f in local_dir_abs.rglob("*") if f.is_file()
        ) / 1024 / 1024
        if mb > size_mb * 0.5:
            print(f"  ✓ 已存在（{mb:.0f}MB）：{local_dir_abs.name}")
            return True

    print(f"  ⏳ 首次使用将下载 {desc} {model_id}（≈{size_mb}MB）")
    print(f"    → 缓存到 {local_dir_abs}")
    try:
        from huggingface_hub import snapshot_download
        import os as _os

        _os.makedirs(local_dir_abs, exist_ok=True)
        commit = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir_abs),
            local_dir_use_symlinks=False,  # 真实文件，不用符号链接
            ignore_patterns=["*.md", "*.txt", "*.py"],  # 不下载 README
            cache_dir=str(ROOT / ".hf_cache"),  # 共享 HF 缓存（避免重复下载）
        )
        # 记录 commit
        manifest[model_id] = {"commit": commit, "size_mb": size_mb}
        save_manifest(manifest)
        print(f"  ✓ 下载完成：{local_dir_abs.name}（commit: {commit[:8]}）")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败：{e}")
        if "Connection" in str(e) or "timeout" in str(e).lower():
            print(f"    提示：国内建议设置环境变量  set HF_ENDPOINT=https://hf-mirror.com")
        return False


def update_config(embedding_path: str | None, reranker_path: str | None):
    """更新 config.json，优先写本地路径（已下载的模型优先）"""
    cfg_file = ROOT / "config.json"
    if not cfg_file.exists():
        print("  ⚠ config.json 不存在，跳过配置更新")
        return
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    changed = False
    if embedding_path:
        cfg["embedding_model_path"] = embedding_path
        changed = True
    if reranker_path:
        cfg["reranker_model_path"] = reranker_path
        changed = True
    if changed:
        cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ config.json 已更新（嵌入模型 → {embedding_path}）")


def main():
    parser = argparse.ArgumentParser(description="下载模型到本地 models/ 目录")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--embedding-small", action="store_true", help="仅下载 bge-small（轻量嵌入）")
    group.add_argument("--embedding-base", action="store_true", help="下载 bge-base（高精度嵌入）")
    group.add_argument("--reranker", action="store_true", help="仅下载重排模型")
    group.add_argument("--all", action="store_true", help="下载全套模型（默认）")
    args = parser.parse_args()

    # 确定下载范围
    models = get_model_info()
    if args.embedding_small:
        models = [m for m in models if m["id"] == "BAAI/bge-small-zh-v1.5"]
    elif args.embedding_base:
        models = [m for m in models if m["id"] == "BAAI/bge-base-zh-v1.5"]
    elif args.reranker:
        models = [m for m in models if m["type"] == "reranker"]
    else:
        pass  # 默认全部

    print(f"\n{'='*60}")
    print(f"  模型下载工具（将保存到 {MODELS_DIR}）")
    print(f"{'='*60}")
    print(f"  HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    print(f"  目标模型: {len(models)} 个\n")

    ok_count = 0
    chosen_embedding = None
    chosen_reranker = None

    for m in models:
        local_abs = MODELS_DIR / m["local_dir"]
        ok = download_model(m["id"], local_abs, desc=m["description"], size_mb=m["size_mb"])
        if ok:
            ok_count += 1
            # 存相对路径（可移植：项目移动目录后仍有效）
            local_rel = f"models/{m['local_dir']}"
            if m["type"] == "embedding" and chosen_embedding is None:
                chosen_embedding = local_rel
            elif m["type"] == "reranker":
                chosen_reranker = local_rel

    print(f"\n{'='*60}")
    if ok_count == len(models):
        print(f"  ✅ 全部下载完成（{ok_count}/{len(models)}）")
        update_config(chosen_embedding, chosen_reranker)
        print(f"\n  接下来：")
        print(f"    1. 重启 GUI（main.py）即可使用本地模型")
        print(f"    2. 向量化时默认使用 bge-small（CPU/GPU 自动）")
        print(f"    3. 若有 GPU，安装 torch CUDA 版以加速：")
        print(f"       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print(f"       （RTX 50 系列必须 cu128+；已有 torch CPU 版需先卸载：pip uninstall torch torchvision torchaudio）")
    else:
        print(f"  ⚠ 部分失败（{ok_count}/{len(models)}），请检查网络后重试")
        print(f"    重试命令：python scripts/download_models.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
