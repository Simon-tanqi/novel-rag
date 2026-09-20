# -*- coding: utf-8 -*-
"""test_import_existing_vector: 覆盖 json 命名兼容性问题"""
import os
import shutil
import pytest
import project_manager as pm_module
from project_manager import ProjectManager


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """隔离项目根到 tmp_path，避免污染真实配置"""
    (tmp_path / "data").mkdir()
    # ProjectManager 从 utils 导入 get_root_dir，需要 patch utils 的引用
    import utils
    monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
    # 同时 patch project_manager 内的引用
    monkeypatch.setattr(pm_module, "get_root_dir", lambda: tmp_path)
    return tmp_path


def _make_vector_db(parent: str, npy_name: str = "embeddings.npy",
                    json_name: str = "metadata.json") -> str:
    """在 parent/vector_db/ 下生成可用的 .npy + .json"""
    vdir = os.path.join(parent, "vector_db")
    os.makedirs(vdir, exist_ok=True)
    # 写一个最小的 npy（1 段 512 维 0 向量）和 metadata（1 条标准格式）
    import numpy as np
    np.save(os.path.join(vdir, npy_name), np.zeros((1, 512), dtype="float32"))
    with open(os.path.join(vdir, json_name), "w", encoding="utf-8") as f:
        import json
        json.dump([{
            "text": "test",
            "chapter": "第1章",
            "chunk_id": 0,
            "chunk_length": 4,
            "coref_prefix": "",
        }], f, ensure_ascii=False)
    return vdir


class TestImportExistingVectorJSONCompat:
    """覆盖「导入向量库无法识别 metadata.json」回归"""

    def test_emb_npy_plus_metadata_json(self, isolated_root):
        """标准项目格式: embeddings.npy + metadata.json"""
        vdb = _make_vector_db(str(isolated_root / "src"),
                              "embeddings.npy", "metadata.json")
        pm = ProjectManager()
        result = pm.import_existing_vector(
            name="novel1",
            source_path="",
            vector_db_path=vdb,
        )
        assert result is not None
        assert result["vector_file"] == "embeddings.npy"
        assert result["metadata_file"] == "metadata.json"
        assert result["status"] == "ready"

    def test_emb_npy_plus_emb_json(self, isolated_root):
        """旧别名: embeddings.npy + embeddings.json"""
        vdb = _make_vector_db(str(isolated_root / "src"),
                              "embeddings.npy", "embeddings.json")
        pm = ProjectManager()
        result = pm.import_existing_vector(
            name="novel2", source_path="", vector_db_path=vdb,
        )
        assert result is not None
        assert result["metadata_file"] == "embeddings.json"

    def test_vectors_npy_plus_metadata_json(self, isolated_root):
        """不同 npy 名 + metadata.json（用户实际场景）"""
        vdb = _make_vector_db(str(isolated_root / "src"),
                              "vectors.npy", "metadata.json")
        pm = ProjectManager()
        result = pm.import_existing_vector(
            name="novel3", source_path="", vector_db_path=vdb,
        )
        assert result is not None
        assert result["vector_file"] == "vectors.npy"
        assert result["metadata_file"] == "metadata.json"

    def test_npy_without_json_fails(self, isolated_root):
        """没 json 应当失败（不是兼容性问题，是真的缺文件）"""
        vdir = os.path.join(str(isolated_root / "src"), "vector_db")
        os.makedirs(vdir, exist_ok=True)
        import numpy as np
        np.save(os.path.join(vdir, "embeddings.npy"), np.zeros((1, 512), dtype="float32"))
        pm = ProjectManager()
        result = pm.import_existing_vector(
            name="novel4", source_path="", vector_db_path=vdir,
        )
        assert result is None  # 应当失败

    def test_no_npy_fails(self, isolated_root):
        """没 npy 应当失败"""
        vdir = os.path.join(str(isolated_root / "src"), "vector_db")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "metadata.json"), "w", encoding="utf-8") as f:
            f.write("[]")
        pm = ProjectManager()
        result = pm.import_existing_vector(
            name="novel5", source_path="", vector_db_path=vdir,
        )
        assert result is None
