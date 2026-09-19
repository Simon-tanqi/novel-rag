# -*- coding: utf-8 -*-
"""测试 2026-09 新增能力：ensure_demo_project 自动注册、cmd_demo 分支、cmd_ingest 同路径跳过。

测试策略：
  - monkeypatch utils.get_root_dir → tmp_path，实现 ProjectManager 与数据目录的完全隔离
  - 不依赖真实嵌入模型或网络；cmd_ingest 全新构建路径用 --embedding "" 走关键词模式
  - 直接检查 config["current_project_id"] 字段，避免 get_current_project() 兜底逻辑掩盖真实状态
"""
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ---- 模块级 fixture：隔离根目录 ----

@pytest.fixture()
def isolated_root(tmp_path, monkeypatch):
    """将 utils.get_root_dir 重定向到 tmp_path，隔离 novel_config.json 与 data/ 目录。

    同时 monkeypatch project_manager.get_root_dir，因为 ProjectManager
    直接 from utils import get_root_dir（模块级绑定）。
    """
    # 在 tmp_path 下预建 data 目录（get_data_dir 会 mkdir，但提前创建更安全）
    (tmp_path / "data").mkdir(exist_ok=True)

    # utils.get_root_dir 被 project_manager 模块级 import 绑定
    import utils
    import project_manager
    monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
    monkeypatch.setattr(project_manager, "get_root_dir", lambda: tmp_path)

    # novel_rag 也 import 了 get_root_dir（但主要用 os.path.dirname(__file__)）
    import novel_rag
    # novel_rag.cmd_demo 用 os.path.dirname(os.path.abspath(__file__)) 取 root
    # 需要monkeypatch使其指向 tmp_path
    monkeypatch.setattr(novel_rag, "__file__", str(tmp_path / "novel_rag.py"))

    return tmp_path


# ===================== ensure_demo_project =====================

class TestEnsureDemoProject:
    """ensure_demo_project：幂等自动注册 + current 兜底"""

    def test_data_exists_auto_registers(self, isolated_root):
        """data/demo 数据存在但未注册 → 自动注册并置 ready"""
        from project_manager import ProjectManager

        # 构造 demo 数据（source + metadata）
        demo_src = isolated_root / "data" / "demo" / "source" / "demo_novel.txt"
        demo_src.parent.mkdir(parents=True, exist_ok=True)
        demo_src.write_text("第一章 测试\n正文内容。\n", encoding="utf-8")

        demo_vdb = isolated_root / "data" / "demo" / "vector_db"
        demo_vdb.mkdir(parents=True, exist_ok=True)
        (demo_vdb / "metadata.json").write_text(
            json.dumps([{"text": "正文", "chapter": "第一章", "chunk_id": 1}],
                       ensure_ascii=False), encoding="utf-8")

        pm = ProjectManager()
        project = pm.get_project_by_name("demo")
        assert project is not None
        assert project["status"] == "ready"
        assert pm.config["current_project_id"] == project["id"]

    def test_already_registered_no_duplicate(self, isolated_root):
        """已注册的 demo 不应重复创建"""
        from project_manager import ProjectManager

        # 先构造数据并首次注册
        demo_src = isolated_root / "data" / "demo" / "source" / "demo_novel.txt"
        demo_src.parent.mkdir(parents=True, exist_ok=True)
        demo_src.write_text("测试内容", encoding="utf-8")

        demo_vdb = isolated_root / "data" / "demo" / "vector_db"
        demo_vdb.mkdir(parents=True, exist_ok=True)
        (demo_vdb / "metadata.json").write_text("[]", encoding="utf-8")

        pm1 = ProjectManager()
        count_after_first = len(pm1.get_all_projects())

        # 再次实例化（模拟重启）
        pm2 = ProjectManager()
        assert len(pm2.get_all_projects()) == count_after_first  # 不增长

    def test_no_data_returns_none(self, isolated_root):
        """data/demo 数据不存在 → 返回 None，不创建空项目"""
        from project_manager import ProjectManager

        pm = ProjectManager()
        assert pm.get_project_by_name("demo") is None
        assert pm.config["current_project_id"] is None

    def test_current_id_fallback_when_missing(self, isolated_root):
        """demo 已注册但 current_project_id 为空 → 兜底设为 demo"""
        from project_manager import ProjectManager

        # 构造数据 + 首次注册
        demo_src = isolated_root / "data" / "demo" / "source" / "demo_novel.txt"
        demo_src.parent.mkdir(parents=True, exist_ok=True)
        demo_src.write_text("测试", encoding="utf-8")

        demo_vdb = isolated_root / "data" / "demo" / "vector_db"
        demo_vdb.mkdir(parents=True, exist_ok=True)
        (demo_vdb / "metadata.json").write_text("[]", encoding="utf-8")

        pm = ProjectManager()
        project = pm.get_project_by_name("demo")

        # 模拟 current 被手工清空
        pm.config["current_project_id"] = None
        pm.save_config()

        # 重新实例化 → ensure_demo_project 应兜底
        pm2 = ProjectManager()
        assert pm2.config["current_project_id"] == project["id"]


# ===================== cmd_demo 分支 =====================

class TestCmdDemo:
    """cmd_demo 三条分支：就绪复用 / force 重建 / 全新构建"""

    @pytest.fixture()
    def demo_ready(self, isolated_root):
        """构造一个已就绪的 demo 项目（含 embeddings.npy）"""
        from project_manager import ProjectManager

        demo_src = isolated_root / "data" / "demo" / "source" / "demo_novel.txt"
        demo_src.parent.mkdir(parents=True, exist_ok=True)
        demo_src.write_text("第一章 测试\n正文。\n", encoding="utf-8")

        demo_vdb = isolated_root / "data" / "demo" / "vector_db"
        demo_vdb.mkdir(parents=True, exist_ok=True)
        # 1 条 4 维向量
        np.save(demo_vdb / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
        (demo_vdb / "metadata.json").write_text(
            json.dumps([{"text": "正文", "chapter": "第一章", "chunk_id": 1}],
                       ensure_ascii=False), encoding="utf-8")

        pm = ProjectManager()
        return pm.get_project_by_name("demo")

    def test_ready_project_prints_and_returns(self, demo_ready, isolated_root, capsys):
        """分支 1：demo 已就绪且有向量 → 打印就绪信息 + QA 卡，不调用 cmd_ingest"""
        import novel_rag
        from novel_rag import cmd_demo

        # 构造 args
        args = MagicMock()
        args.force = False
        args.no_embedding = False

        # spy cmd_ingest 确保不被调用
        with patch.object(novel_rag, "cmd_ingest") as mock_ingest:
            cmd_demo(args)
            mock_ingest.assert_not_called()

        captured = capsys.readouterr()
        assert "demo 项目已就绪" in captured.out
        assert "语义向量库" in captured.out  # 有 embeddings.npy → 语义模式

    def test_ready_keyword_mode_prints_hint(self, isolated_root, capsys):
        """分支 1 变体：demo 就绪但无 embeddings.npy → 关键词模式 + 提示安装"""
        from project_manager import ProjectManager
        import novel_rag
        from novel_rag import cmd_demo

        # 构造无向量的 demo（只有 metadata）
        demo_src = isolated_root / "data" / "demo" / "source" / "demo_novel.txt"
        demo_src.parent.mkdir(parents=True, exist_ok=True)
        demo_src.write_text("测试", encoding="utf-8")

        demo_vdb = isolated_root / "data" / "demo" / "vector_db"
        demo_vdb.mkdir(parents=True, exist_ok=True)
        (demo_vdb / "metadata.json").write_text("[]", encoding="utf-8")

        pm = ProjectManager()
        assert pm.get_project_by_name("demo") is not None

        args = MagicMock()
        args.force = False
        args.no_embedding = False

        with patch.object(novel_rag, "cmd_ingest") as mock_ingest:
            cmd_demo(args)
            mock_ingest.assert_not_called()

        captured = capsys.readouterr()
        assert "关键词检索模式" in captured.out

    def test_force_deletes_and_rebuilds(self, demo_ready, isolated_root, capsys):
        """分支 2：--force 删除现有 demo 并重建

        --force 删除旧项目 → 写入原文 → ensure_demo_project 无数据返回 None
        → cmd_demo 调用 cmd_ingest 走全新构建。这里 mock cmd_ingest
        仅验证被调用（真实构建路径由 TestCmdIngestSamePathSkip 覆盖）。
        """
        import novel_rag
        from novel_rag import cmd_demo
        from project_manager import ProjectManager

        old_id = demo_ready["id"]

        args = MagicMock()
        args.force = True
        args.no_embedding = True

        # mock cmd_ingest：记录调用参数，不真实执行
        captured_args = {}
        def fake_ingest(a):
            captured_args["file"] = a.file
            captured_args["name"] = a.name
            captured_args["embedding"] = a.embedding

        with patch.object(novel_rag, "cmd_ingest", side_effect=fake_ingest) as mock_ingest:
            cmd_demo(args)
            mock_ingest.assert_called_once()

        captured = capsys.readouterr()
        assert "--force" in captured.out or "重建" in captured.out

        # 旧项目已删除
        pm = ProjectManager()
        assert pm.get_project(old_id) is None

        # cmd_ingest 收到的 name 应为 demo
        assert captured_args["name"] == "demo"
        assert captured_args["embedding"] == ""  # --no-embedding → 空串

    def test_fresh_build_calls_ingest(self, isolated_root, capsys):
        """分支 3：全新环境（无 data/demo）→ 生成原文 + 调用 cmd_ingest"""
        import novel_rag
        from novel_rag import cmd_demo

        args = MagicMock()
        args.force = False
        args.no_embedding = True

        with patch.object(novel_rag, "cmd_ingest") as mock_ingest:
            cmd_demo(args)
            mock_ingest.assert_called_once()

        # 验证原文已写入
        demo_file = isolated_root / "data" / "demo" / "source" / "demo_novel.txt"
        assert demo_file.is_file()
        assert len(demo_file.read_text(encoding="utf-8")) > 100  # 有实质内容


# ===================== cmd_ingest 同路径跳过 =====================

class TestCmdIngestSamePathSkip:
    """cmd_ingest：源文件已在项目源目录时跳过复制（防同文件 copy2 报错）"""

    def test_same_path_skips_copy(self, isolated_root, capsys):
        """源文件就在目标 source 目录 → 打印跳过，不报错"""
        import novel_rag
        from novel_rag import cmd_ingest

        # 在 data/demo/source/ 下放一个 txt（与项目名同路径）
        src_dir = isolated_root / "data" / "demo" / "source"
        src_dir.mkdir(parents=True, exist_ok=True)
        src_file = src_dir / "demo_novel.txt"
        src_file.write_text("第一章 测试\n正文。\n" * 20, encoding="utf-8")

        args = MagicMock()
        args.file = str(src_file)
        args.name = "demo"
        args.chunk_size = 500
        args.overlap = 50
        args.rules = None
        args.words = None
        args.embedding = ""  # 纯关键词模式

        # cmd_ingest 会走完整流程（清洗→切片→向量化→注册）
        # 但我们不希望真的跑向量化，只验证跳过复制分支
        # 所以 patch build_vector_index_from_file 避免副作用
        with patch.object(novel_rag, "build_vector_index_from_file", return_value=True):
            cmd_ingest(args)

        captured = capsys.readouterr()
        assert "跳过复制" in captured.out

        # 项目已注册
        from project_manager import ProjectManager
        pm = ProjectManager()
        project = pm.get_project_by_name("demo")
        assert project is not None
        # args.embedding="" 表示无嵌入模型，且此处 patch 掉的建库未产出 embeddings.npy，
        # 因此状态应为 keyword_only（修复前会无条件写成 ready，造成"向量检索不可用"的假象）
        assert project["status"] == "keyword_only"
