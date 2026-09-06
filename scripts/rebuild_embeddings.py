"""
rebuild_embeddings.py — 重新生成 embeddings.npy，不重新切片

适用场景：metadata.json 存在但 embeddings.npy 缺失（典型：ingest 时嵌入模型
加载失败，自动降级到关键词模式，status 被错误地设成 ready）

用法（PowerShell）：
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python scripts/rebuild_embeddings.py
"""
import os
import sys
import time

# 路径设置
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import project_manager
from step2_split_embed import build_vector_index_from_file

# 1) 找目标项目（命令行参数 > 第一个非 demo 项目 > 提示用户）
target_name = sys.argv[1] if len(sys.argv) > 1 else None

pm = project_manager.ProjectManager()
if target_name:
    proj = pm.get_project_by_name(target_name)
else:
    # 默认：找 metadata.json 在但 npy 不在的项目
    candidates = []
    for p in pm.config["projects"]:
        vd = p.get("vector_db_path", "")
        if not vd or not os.path.isdir(vd):
            continue
        meta_ok = os.path.exists(os.path.join(vd, "metadata.json"))
        npy_ok = os.path.exists(os.path.join(vd, "embeddings.npy"))
        if meta_ok and not npy_ok:
            candidates.append(p)
    if not candidates:
        print("✓ 没有需要修复的项目（所有项目 embeddings.npy 都已就绪）")
        sys.exit(0)
    if len(candidates) > 1:
        print("找到多个待修复项目，请用参数指定：")
        for p in candidates:
            print(f"  - {p['name']!r}")
        sys.exit(1)
    proj = candidates[0]

if not proj:
    print(f"✗ 找不到项目: {target_name!r}")
    sys.exit(1)

print(f"目标项目: {proj['name']}")
print(f"  source:     {proj['source_path']}")
print(f"  cleaned:    {proj.get('cleaned_path', '')}")
print(f"  vector_db:  {proj['vector_db_path']}")

cleaned = proj.get("cleaned_path", "")
if not cleaned or not os.path.exists(cleaned):
    print(f"✗ cleaned 文件不存在: {cleaned!r}")
    sys.exit(1)

# 2) 取嵌入模型（与 ingest 时同源）
import config_manager
cfg_mgr = config_manager.ConfigManager()
embedding_path = cfg_mgr.get("embedding_model_path", "") or "BAAI/bge-small-zh-v1.5"
print(f"\n嵌入模型: {embedding_path}")
print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '(未设置)')}\n")

# 3) 跑向量化（会重读 cleaned → 重新切片 → encode → 写 npy + 覆盖 meta）
#    切片是确定性的（chunk_size/overlap 固定），所以 meta 与原版等价
def progress(step, total, msg, pct=0):
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:3d}%  {msg[:50]}", end="", flush=True)

t0 = time.time()
ok = build_vector_index_from_file(
    input_file=cleaned,
    chunk_size=proj.get("chunk_size", 500),
    overlap=proj.get("overlap", 50),
    output_dir=proj["vector_db_path"],
    embedding_model_path=embedding_path,
    progress_callback=progress,
)
print()  # 换行
elapsed = time.time() - t0

if ok:
    # 4) 验证 npy
    import numpy as np
    npy_path = os.path.join(proj["vector_db_path"], "embeddings.npy")
    arr = np.load(npy_path)
    meta_path = os.path.join(proj["vector_db_path"], "metadata.json")
    import json
    meta = json.load(open(meta_path, encoding="utf-8"))
    print(f"\n✅ 重建完成 ({elapsed:.1f}s)")
    print(f"   embeddings.npy: {arr.shape}  ({arr.dtype})  L2 归一化: {np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-3)}")
    print(f"   metadata.json:  {len(meta)} 段")
    print(f"\n现在可以重新启动 GUI 提问，应该会获得真实向量检索结果。")
else:
    print(f"\n✗ 重建失败（{elapsed:.1f}s）")
    print(f"  常见原因:")
    print(f"  1) HF_ENDPOINT 未设：$env:HF_ENDPOINT='https://hf-mirror.com'")
    print(f"  2) 网络问题：直连 huggingface.co 超时")
    print(f"  3) sentence-transformers 未装：pip install sentence-transformers")
    sys.exit(1)
