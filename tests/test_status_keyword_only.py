"""测试 get_project_status 在 npy 缺失时返回 keyword_only，以及 is_keyword_only helper"""
import os
import sys
import shutil
import json
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils  # 用于 monkeypatch utils.get_root_dir
import project_manager as pm_module
from project_manager import ProjectManager


@pytest.fixture
def isolated_root(monkeypatch, tmp_path):
    """monkeypatch utils.get_root_dir 指向 tmp_path（project_manager.create_project
    内部走 utils.get_data_dir -> utils.get_root_dir 链）"""
    monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
    monkeypatch.setattr(pm_module, "get_root_dir", lambda: tmp_path)
    return tmp_path


def _make_project(isolated_root, name="test_proj"):
    """构造一个完整项目：source + cleaned + metadata.json，npy 缺失"""
    proj_dir = isolated_root / "data" / name
    (proj_dir / "source").mkdir(parents=True)
    (proj_dir / "cleaned").mkdir(parents=True)
    (proj_dir / "vector_db").mkdir(parents=True)
    # 源文件
    (proj_dir / "source" / f"{name}.txt").write_text("测试文本", encoding="utf-8")
    # cleaned
    (proj_dir / "cleaned" / "cleaned.txt").write_text("测试清洗后文本", encoding="utf-8")
    # metadata.json（3 段）
    meta = [
        {"text": f"第{i}段文本", "chapter": "第一章", "chunk_id": i, "chunk_length": 10, "coref_prefix": ""}
        for i in range(1, 4)
    ]
    (proj_dir / "vector_db" / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    # 故意不写 embeddings.npy
    return proj_dir


def test_keyword_only_when_npy_missing(isolated_root):
    """npy 缺失时 status 返回 keyword_only（不是 ready）"""
    _make_project(isolated_root)
    pm = ProjectManager()
    proj = pm.create_project(
        name="test_proj",
        source_path=str(isolated_root / "data" / "test_proj" / "source" / "test_proj.txt"),
        chunk_size=500,
        overlap=50,
    )
    # 模拟 status="ready" 但实际是 keyword_only
    status = pm.get_project_status(proj["id"])
    assert status == "keyword_only", f"期望 keyword_only，实际 {status!r}"
    assert pm.is_keyword_only(proj["id"]) is True


def test_ready_when_both_files_exist(isolated_root):
    """npy 和 meta 都在时 status 返回 ready"""
    proj_dir = _make_project(isolated_root)
    import numpy as np
    np.save(proj_dir / "vector_db" / "embeddings.npy", np.zeros((3, 4), dtype=np.float32))
    pm = ProjectManager()
    proj = pm.create_project(
        name="test_proj",
        source_path=str(isolated_root / "data" / "test_proj" / "source" / "test_proj.txt"),
        chunk_size=500,
        overlap=50,
    )
    assert pm.get_project_status(proj["id"]) == "ready"
    assert pm.is_keyword_only(proj["id"]) is False
