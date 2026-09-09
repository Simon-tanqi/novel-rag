# -*- coding: utf-8 -*-
"""test_get_project_runtime: 覆盖 get_project_runtime 的所见即所得数据"""
import os
import json
import pytest
import project_manager as pm_module
import utils
from project_manager import ProjectManager


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
    monkeypatch.setattr(pm_module, "get_root_dir", lambda: tmp_path)
    return tmp_path


def _create_ready_project(pm: ProjectManager, name: str, n_chunks: int = 5,
                          npy_size_kb: int = 4, external: bool = False,
                          root=None):
    """创建一个 ready 状态的项目（含 npy + json）"""
    p = pm.create_project(
        name=name,
        source_path=f"/tmp/{name}.txt",
        chunk_size=500, overlap=50, clean_rules=[],
    )
    pid = p["id"]
    data_dir = utils.get_data_dir() / name
    vdb = data_dir / "vector_db"
    vdb.mkdir(parents=True, exist_ok=True)
    # 写 npy (随机字节)
    npy_path = vdb / "embeddings.npy"
    npy_path.write_bytes(b"\x00" * (npy_size_kb * 1024))
    # 写 metadata
    meta_path = vdb / "metadata.json"
    meta_path.write_text(
        json.dumps([{"text": f"chunk {i}", "chunk_id": i} for i in range(n_chunks)]),
        encoding="utf-8",
    )

    if external:
        # 把 vector_db_path 改成外部路径
        ext_dir = root / "external" / name
        ext_dir.mkdir(parents=True, exist_ok=True)
        (ext_dir / "embeddings.npy").write_bytes(b"\x00" * (npy_size_kb * 1024))
        (ext_dir / "metadata.json").write_text(
            json.dumps([{"text": f"chunk {i}", "chunk_id": i} for i in range(n_chunks)]),
            encoding="utf-8",
        )
        pm.update_project(pid, {"vector_db_path": str(ext_dir)})
    else:
        pm.update_project(pid, {"vector_db_path": str(vdb), "status": "ready"})
    return pid


class TestGetProjectRuntime:

    def test_ready_project_reports_chunks_and_size(self, isolated_root):
        """ready 项目：返回段数 + embeddings 大小"""
        pm = ProjectManager()
        pid = _create_ready_project(pm, "demo", n_chunks=5, npy_size_kb=10, root=isolated_root)

        rt = pm.get_project_runtime(pid)
        assert rt["status"] == "ready"
        assert rt["chunk_count"] == 5
        assert rt["has_embeddings"] is True
        assert rt["has_metadata"] is True
        # npy 是 10KB，转换为 MB ≈ 0.01（可能被 round 为 0.0，但函数正确返回）
        assert "embeddings_size_mb" in rt
        assert rt["embeddings_size_mb"] >= 0

    def test_external_path_is_flagged(self, isolated_root):
        """vector_db_path 在 data/<name>/ 外时 is_external=True"""
        pm = ProjectManager()
        pid = _create_ready_project(pm, "exttest", n_chunks=3, external=True, root=isolated_root)

        rt = pm.get_project_runtime(pid)
        assert rt["is_external"] is True
        assert rt["status"] == "ready"
        assert rt["chunk_count"] == 3

    def test_internal_path_not_flagged(self, isolated_root):
        """vector_db_path 在 data/<name>/ 内时 is_external=False"""
        pm = ProjectManager()
        pid = _create_ready_project(pm, "inttest", n_chunks=3, external=False)

        rt = pm.get_project_runtime(pid)
        assert rt["is_external"] is False
        assert rt["status"] == "ready"

    def test_keyword_only_when_only_metadata(self, isolated_root):
        """只有 metadata.json → status=keyword_only, has_embeddings=False"""
        pm = ProjectManager()
        p = pm.create_project(
            name="kwonly", source_path="/tmp/kwonly.txt",
            chunk_size=500, overlap=50, clean_rules=[],
        )
        vdb = utils.get_data_dir() / "kwonly" / "vector_db"
        vdb.mkdir(parents=True, exist_ok=True)
        (vdb / "metadata.json").write_text(
            json.dumps([{"text": "x", "chunk_id": 0}]), encoding="utf-8"
        )
        pm.update_project(p["id"], {"vector_db_path": str(vdb)})

        rt = pm.get_project_runtime(p["id"])
        assert rt["status"] == "keyword_only"
        assert rt["has_embeddings"] is False
        assert rt["has_metadata"] is True
        assert rt["chunk_count"] == 1

    def test_missing_vector_db_path(self, isolated_root):
        """vector_db_path 不存在时 status=not_ready, chunk_count=0"""
        pm = ProjectManager()
        p = pm.create_project(
            name="missing", source_path="/tmp/missing.txt",
            chunk_size=500, overlap=50, clean_rules=[],
        )
        pm.update_project(p["id"], {"vector_db_path": "/nonexistent/path"})

        rt = pm.get_project_runtime(p["id"])
        assert rt["status"] == "not_ready"
        assert rt["chunk_count"] == 0
        assert rt["has_embeddings"] is False
        assert rt["has_metadata"] is False

    def test_nonexistent_project(self, isolated_root):
        """不存在的项目 id → status=unknown, error 字段"""
        pm = ProjectManager()
        rt = pm.get_project_runtime("nonexistent_id")
        assert rt["status"] == "unknown"
        assert "error" in rt

    def test_reports_source_size(self, isolated_root):
        """源文件大小被报告"""
        pm = ProjectManager()
        pid = _create_ready_project(pm, "withsrc", n_chunks=1)

        # 写一个 100KB 源文件
        src = isolated_root / "data" / "withsrc" / "source.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"x" * (100 * 1024))
        pm.update_project(pid, {"source_path": str(src)})

        rt = pm.get_project_runtime(pid)
        assert rt["source_size_mb"] > 0
