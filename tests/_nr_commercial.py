# -*- coding: utf-8 -*-
"""
tests/_nr_commercial.py — 商用验收测试公共脚手架（非测试文件，不参与收集）

提供：项目路径常量、真实语料只读访问、确定性嵌入替身、离线检索器构造、
最小向量库构造、CLI 子进程执行、数值指标落盘（供报告引用）。

约束：
- 只读项目真实数据，绝不写入 data/ 与仓库；
- 所有产物写入 pytest tmp_path；指标 JSONL 写入环境变量
  NR_COMMERCIAL_METRICS 指定路径（未设置则仅打印，不落盘）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PROD_PROJECT = "绝世主宰"
PROD_CORPUS = DATA_DIR / PROD_PROJECT / "cleaned" / "cleaned.txt"
PROD_VDB = DATA_DIR / PROD_PROJECT / "vector_db"
DEMO_VDB = DATA_DIR / "demo" / "vector_db"

VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable

_METRICS_ENV = "NR_COMMERCIAL_METRICS"


# ==================== 数值指标落盘 ====================

def record_metric(name: str, value, unit: str = "", note: str = "") -> None:
    """记录一条实测数值指标（供验收报告引用真实数字）。"""
    line = json.dumps(
        {"metric": name, "value": value, "unit": unit, "note": note},
        ensure_ascii=False,
    )
    print(f"[METRIC] {line}")
    path = os.environ.get(_METRICS_ENV, "")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ==================== 确定性嵌入替身 ====================

class FakeEmbeddingModel:
    """字符哈希袋向量替身：同文本必同向量，全离线、秒级。

    兼容 step2 / retriever 调用签名：
        encode(texts, normalize_embeddings=..., show_progress_bar=..., convert_to_numpy=...)
    """

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.encode_calls: list = []

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for ch in str(text):
            v[ord(ch) % self.dim] += 1.0
        if not v.any():
            v[0] = 1.0
        return v

    def encode(self, texts, normalize_embeddings=False,
               show_progress_bar=False, **kwargs):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        self.encode_calls.append(len(texts))
        arr = np.stack([self._vec(t) for t in texts]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr[0] if single else arr

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


def install_fake_embedding(monkeypatch, dim: int = 16):
    """把 step2 的 load_embedding_model 换成替身（离线建库）。"""
    import step2_split_embed as step2
    fake = FakeEmbeddingModel(dim=dim)
    monkeypatch.setattr(step2, "load_embedding_model",
                        lambda *a, **k: (fake, "cpu"))
    return fake


# ==================== 离线检索器构造 ====================

def make_retriever(vector_path, embedding_model=None, reranker_model=None):
    """绕过 __init__ 的重模型加载，直接构造检索器（离线、快）。"""
    from rag_retriever import RAGRetriever
    r = RAGRetriever.__new__(RAGRetriever)
    r.vector_path = str(vector_path)
    r.reranker_model_path = None
    r.embedding_model_path = None
    r.vector_file = None
    r.metadata_file = None
    r.documents = []
    r.parent_chunks = {}
    r.embeddings = None
    r.texts = []
    r.reranker_model = reranker_model
    r.embedding_model = embedding_model
    r._embedding_loaded = True
    r._reranker_loaded = True
    r.last_vector_status = ""
    r.last_fusion_mode = ""
    r.last_retrieval_queries = []
    r._last_vector_error = ""
    r.load_documents()
    return r


def write_library(vd: Path, texts: list, vectors=None, metadata=None,
                  chapters=None, write_npy: bool = True,
                  dim: int = 16) -> Path:
    """在 tmp 目录构造最小向量库（标准 embeddings.npy + metadata.json）。"""
    vd = Path(vd)
    vd.mkdir(parents=True, exist_ok=True)
    chapters = chapters or [f"第{i + 1}章" for i in range(len(texts))]
    if metadata is None:
        metadata = [
            {"text": t, "chapter": chapters[i], "chunk_id": i}
            for i, t in enumerate(texts)
        ]
    if write_npy:
        if vectors is None:
            fake = FakeEmbeddingModel(dim=dim)
            vectors = fake.encode(texts, normalize_embeddings=True)
        np.save(vd / "embeddings.npy", np.asarray(vectors, dtype=np.float32))
    with open(vd / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    return vd


def build_tiny_index(tmp_dir: Path, text: str, monkeypatch, dim: int = 16,
                     name: str = "vdb") -> tuple:
    """用替身模型离线建库，返回 (ok, output_dir, fake_model)。"""
    import step2_split_embed as step2
    fake = install_fake_embedding(monkeypatch, dim=dim)
    out = Path(tmp_dir) / name
    ok = step2.build_vector_index(
        text=text, output_dir=str(out), embedding_model_path="fake",
        book_id="b", book_title="测试书",
    )
    return ok, out, fake


# ==================== CLI 子进程 ====================

def run_cli(args: list, timeout: int = 300, extra_env: dict = None,
            cwd: str = None) -> subprocess.CompletedProcess:
    """在项目根目录以项目虚拟环境执行 CLI（真实端到端路径）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("DEEPSEEK_API_KEY", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [PYTHON] + args, cwd=cwd or str(ROOT), env=env,
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout,
    )


def run_pyscript(code: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """在项目根目录执行一段隔离的 python 代码（用于不可侵入的 CLI 路径验证）。"""
    return run_cli(["-c", code], timeout=timeout)


def percentile(values: list, p: float) -> float:
    """简单百分位（线性插值）。"""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def timed(fn, *a, **k):
    """返回 (结果, 耗时秒)。"""
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0
