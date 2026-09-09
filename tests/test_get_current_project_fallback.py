# -*- coding: utf-8 -*-
"""test_get_current_project_fallback: 覆盖 get_current_project 自动选 ready 项目的逻辑"""
import os
import pytest
import project_manager as pm_module
from project_manager import ProjectManager


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    import utils
    monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
    monkeypatch.setattr(pm_module, "get_root_dir", lambda: tmp_path)
    return tmp_path


def _make_project(pm: ProjectManager, name: str, status: str) -> str:
    """创建一个项目并返回 id"""
    p = pm.create_project(
        name=name,
        source_path=f"/tmp/{name}.txt",
        chunk_size=500, overlap=50, clean_rules=[],
    )
    if status:
        pm.update_project(p["id"], {"status": status})
    return p["id"]


class TestGetCurrentProjectFallback:
    """get_current_project 智能 fallback 到 ready 项目"""

    def test_current_id_ready_returns_it(self, isolated_root):
        """current_id 指向 ready → 返回它"""
        pm = ProjectManager()
        id_a = _make_project(pm, "a_ready", "ready")
        id_b = _make_project(pm, "b_ready", "ready")
        pm.config["current_project_id"] = id_a
        p = pm.get_current_project()
        assert p["id"] == id_a
        assert p["name"] == "a_ready"

    def test_current_id_not_ready_falls_back_to_recent_ready(self, isolated_root):
        """current_id 指向 cleaned → fallback 到最近 ready"""
        pm = ProjectManager()
        _make_project(pm, "old_cleaned", "cleaned")  # 老 cleaned
        id_b = _make_project(pm, "good_ready", "ready")  # 后建的 ready
        # current 指向老的 cleaned
        pm.config["current_project_id"] = pm.config["projects"][0]["id"]
        p = pm.get_current_project()
        assert p["id"] == id_b  # ← 应该 fallback 到这个 ready

    def test_current_id_none_falls_back_to_recent_ready(self, isolated_root):
        """current_id 为空 → fallback 到最近 ready"""
        pm = ProjectManager()
        _make_project(pm, "old_cleaned", "cleaned")
        id_b = _make_project(pm, "good_ready", "ready")
        pm.config["current_project_id"] = None
        p = pm.get_current_project()
        assert p["id"] == id_b

    def test_no_ready_falls_back_to_last(self, isolated_root):
        """没任何 ready → 兜底返回最后一个（保持向后兼容）"""
        pm = ProjectManager()
        _make_project(pm, "p1", "cleaned")
        _make_project(pm, "p2", "created")
        pm.config["current_project_id"] = None
        p = pm.get_current_project()
        assert p["name"] == "p2"  # ← 兜底行为

    def test_empty_projects_returns_none(self, isolated_root):
        """没任何项目 → 返回 None"""
        pm = ProjectManager()
        assert pm.get_current_project() is None
