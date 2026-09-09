# -*- coding: utf-8 -*-
"""测试 utils.resolve_embedding_model 智能识别逻辑

覆盖：
- 绝对路径存在 → local
- 相对路径按 cwd 解析
- 相对路径按项目根解析（关键：GUI 启动时 cwd 不在项目根）
- HF ID 形式（如 BAAI/bge-small-zh-v1.5）但项目根/models/ 有同名 → local
- HF ID 无本地 → hub
- 空/None → None
- 异常路径 → hub（兜底）
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import resolve_embedding_model, model_id_to_dirname, get_root_dir


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    """重定向 get_root_dir 到 tmp_path，让相对路径解析走测试目录"""
    import utils
    monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
    return tmp_path


def test_resolve_empty_returns_none():
    assert resolve_embedding_model(None) is None
    assert resolve_embedding_model("") is None
    assert resolve_embedding_model("   ") is None


def test_resolve_absolute_path_exists(tmp_path):
    """绝对路径且目录存在 → local"""
    d = tmp_path / "my_model"
    d.mkdir()
    result = resolve_embedding_model(str(d))
    assert result == ("local", str(d))


def test_resolve_absolute_path_not_exists(tmp_path):
    """绝对路径但目录不存在 → hub（fallback，让调用方下载）"""
    d = tmp_path / "nonexistent_model"
    result = resolve_embedding_model(str(d))
    assert result is None or result[0] == "hub"


def test_resolve_relative_via_project_root(isolated_root, monkeypatch):
    """关键场景：GUI 启动时 cwd 不在项目根，相对路径应能通过项目根解析"""
    # 把 cwd 改到无关目录（如系统临时目录）
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))  # tests 目录
    # 在项目根（tmp_path）下建 models/foo
    target = isolated_root / "models" / "foo"
    target.mkdir(parents=True)

    result = resolve_embedding_model("models/foo")
    assert result is not None
    assert result[0] == "local"
    assert Path(result[1]).resolve() == target.resolve()


def test_resolve_relative_via_cwd(tmp_path, monkeypatch):
    """相对路径在 cwd 解析（兼容 import 时的 cwd 场景）"""
    # 在 cwd 建目录
    target = tmp_path / "local_model"
    target.mkdir()
    monkeypatch.chdir(tmp_path)

    # 不 monkeypatch get_root_dir，解析时应能命中 cwd 下的目录
    result = resolve_embedding_model("local_model")
    assert result is not None
    assert result[0] == "local"


def test_resolve_hf_id_with_local_fallback(isolated_root):
    """HF ID 形式 BAAI/bge-small-zh-v1.5，且项目根/models/bge-small-zh-v1.5 存在 → local"""
    target = isolated_root / "models" / "bge-small-zh-v1.5"
    target.mkdir(parents=True)
    result = resolve_embedding_model("BAAI/bge-small-zh-v1.5")
    assert result is not None
    assert result[0] == "local"
    assert Path(result[1]).resolve() == target.resolve()


def test_resolve_hf_id_without_local_returns_hub(isolated_root):
    """HF ID 形式但本地无 → hub（让调用方下载到 models/）"""
    result = resolve_embedding_model("BAAI/bge-small-zh-v1.5")
    assert result is not None
    assert result[0] == "hub"
    assert result[1] == "BAAI/bge-small-zh-v1.5"


def test_resolve_hf_id_with_dashed_name(isolated_root):
    """org/name 形式，name 含连字符/点/下划线都能匹配本地 fallback"""
    target = isolated_root / "models" / "bge-reranker-base"
    target.mkdir(parents=True)
    result = resolve_embedding_model("BAAI/bge-reranker-base")
    assert result is not None
    assert result[0] == "local"
    assert Path(result[1]).resolve() == target.resolve()


def test_resolve_hf_id_not_matching_format(tmp_path, monkeypatch):
    """不像 org/name 的字符串（如单段 'foo'）→ hub 兜底"""
    monkeypatch.setattr("utils.get_root_dir", lambda: tmp_path)
    result = resolve_embedding_model("foo")
    # 'foo' 不像 org/name（没有 /），但也不存在 → hub
    assert result is not None
    # 关键：不返回 None
    assert result[0] in ("hub", "local")


def test_model_id_to_dirname():
    assert model_id_to_dirname("BAAI/bge-small-zh-v1.5") == "bge-small-zh-v1.5"
    assert model_id_to_dirname("bge-small-zh-v1.5") == "bge-small-zh-v1.5"
    assert model_id_to_dirname("sentence-transformers/all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"
