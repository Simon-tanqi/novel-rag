"""
project_manager.py — 小说项目管理模块
负责 novel_config.json 的读写、项目的增删改查

配置文件结构:
{
  "projects": [
    {
      "id": "project_001",
      "name": "三体",
      "source_path": "./data/三体/source.txt",
      "cleaned_path": "./data/三体/cleaned.txt",
      "vector_db_path": "./data/三体/vector_db/",
      "vector_file": "遮天_20260610.npy",  // 向量化文件名（可选）
      "metadata_file": "遮天_20260610.json",  // 元数据文件名（可选）
      "chunk_size": 672,
      "overlap": 50,
      "clean_rules": ["去除空行", "去除页码"],
      "custom_dirty_words": ["手打", "支持正版"],
      "created_at": "2026-07-25 10:00:00",
      "last_used": "2026-07-25 14:30:00"
    }
  ],
  "current_project_id": "project_001"
}
"""
import json
import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from utils import (
    get_root_dir,
    get_data_dir,
    get_project_dir,
    get_vector_db_path,
    to_relative_path,
    to_absolute_path,
    generate_project_id,
    format_timestamp
)


class ProjectManager:
    """小说项目管理器"""

    # 内置 demo 项目参数（与 novel_rag.py demo 命令保持一致）
    _DEMO_NAME = "demo"
    _DEMO_SOURCE_FILE = "demo_novel.txt"
    _DEMO_CHUNK_SIZE = 672
    _DEMO_OVERLAP = 50
    # 与 step1_clean.RULE_FUNCTIONS 全量规则一致（预置数据已按此清洗）
    _DEMO_CLEAN_RULES = ["去除空行", "去除页码", "去除广告", "合并段落", "修复编码", "去除拼音"]

    def __init__(self):
        self.config_file = get_root_dir() / "novel_config.json"
        self.config = self._load_config()
        # 仓库预置 demo 数据存在但未注册（如 clone 后首次运行）→ 自动注册
        self.ensure_demo_project()

    def ensure_demo_project(self) -> Optional[Dict]:
        """确保内置 demo 项目已注册（幂等，可安全重复调用）。

        - 已注册 → 直接返回现有记录
        - data/demo 数据存在但未注册（clone 仓库后首跑）→ 自动注册并置 ready，不重建数据
        - 数据不存在（全新环境）→ 返回 None，由 novel_rag.py demo 命令完整构建
        """
        existing = self.get_project_by_name(self._DEMO_NAME)
        if existing:
            # 兜底：配置中无当前项目（如被手工精简）→ 默认选中 demo
            if not self.config.get("current_project_id"):
                self.set_current_project(existing["id"])
            return existing

        demo_root = get_root_dir() / "data" / self._DEMO_NAME
        source_file = demo_root / "source" / self._DEMO_SOURCE_FILE
        meta_file = demo_root / "vector_db" / "metadata.json"
        if not source_file.is_file() or not meta_file.is_file():
            return None

        print(f"? 检测到预置 demo 数据，自动注册项目「{self._DEMO_NAME}」...")
        project = self.create_project(
            name=self._DEMO_NAME,
            source_path=str(source_file),
            chunk_size=self._DEMO_CHUNK_SIZE,
            overlap=self._DEMO_OVERLAP,
            clean_rules=list(self._DEMO_CLEAN_RULES),
        )
        self.update_project(project["id"], {"status": "ready"})
        # 首次启动无当前项目时，默认选中 demo（GUI/CLI 开箱即用）
        if not self.config.get("current_project_id"):
            self.set_current_project(project["id"])
        return project

    def _load_config(self) -> Dict:
        """加载配置文件（磁盘存相对路径，加载后转绝对路径）"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                # 路径字段：相对 -> 绝对（项目移动目录后仍可正常使用）
                for p in cfg.get("projects", []):
                    for key in ("source_path", "cleaned_path", "vector_db_path"):
                        if p.get(key):
                            p[key] = to_absolute_path(p[key])
                return cfg
            except (json.JSONDecodeError, IOError) as e:
                print(f"加载项目配置失败: {e}")
                return self._get_default_config()
        return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "projects": [],
            "current_project_id": None
        }

    def save_config(self):
        """保存配置文件（内存中绝对路径，写入磁盘前转相对路径）"""
        try:
            # 深拷贝避免修改内存中的绝对路径
            import copy
            cfg_to_save = copy.deepcopy(self.config)
            for p in cfg_to_save.get("projects", []):
                for key in ("source_path", "cleaned_path", "vector_db_path"):
                    if p.get(key):
                        p[key] = to_relative_path(p[key])
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(cfg_to_save, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"保存项目配置失败: {e}")
            raise

    # ===================== 项目CRUD =====================

    def create_project(
        self,
        name: str,
        source_path: str,
        chunk_size: int = 672,
        overlap: int = 50,
        clean_rules: Optional[List[str]] = None,
        custom_dirty_words: Optional[List[str]] = None
    ) -> Dict:
        """
        创建新项目

        Args:
            name: 小说名称
            source_path: 源文件路径
            chunk_size: 切片硬上限（字符数，缺省 672 ≈ 480 token，最小 210）
            overlap: 相邻片段重叠字符数（默认 50）
            clean_rules: 清洗规则列表
            custom_dirty_words: 自定义脏数据关键词

        Returns:
            新项目配置字典
        """
        project_id = generate_project_id()
        project_dir = get_project_dir(name)

        # 创建目录结构
        source_dir = project_dir / "source"
        source_dir.mkdir(exist_ok=True)

        cleaned_dir = project_dir / "cleaned"
        cleaned_dir.mkdir(exist_ok=True)

        vector_db_path = get_vector_db_path(name)

        # 构建项目配置（使用相对路径）
        project_config = {
            "id": project_id,
            "name": name,
            "source_path": str(source_dir / os.path.basename(source_path)),
            "cleaned_path": str(cleaned_dir / "cleaned.txt"),
            "vector_db_path": vector_db_path,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "clean_rules": clean_rules or [],
            "custom_dirty_words": custom_dirty_words or [],
            "created_at": format_timestamp(),
            "last_used": format_timestamp(),
            "status": "created"  # created → cleaning → embedding → ready → failed
        }

        self.config["projects"].append(project_config)
        self.save_config()

        return project_config

    def get_project(self, project_id: str) -> Optional[Dict]:
        """根据ID获取项目"""
        for project in self.config["projects"]:
            if project["id"] == project_id:
                return project
        return None

    def get_project_by_name(self, name: str) -> Optional[Dict]:
        """根据名称获取项目"""
        for project in self.config["projects"]:
            if project["name"] == name:
                return project
        return None

    def get_all_projects(self) -> List[Dict]:
        """获取所有项目列表"""
        return self.config["projects"]

    def update_project(self, project_id: str, updates: Dict) -> bool:
        """更新项目配置"""
        for i, project in enumerate(self.config["projects"]):
            if project["id"] == project_id:
                self.config["projects"][i].update(updates)
                self.save_config()
                return True
        return False

    def delete_project(self, project_id: str) -> bool:
        """删除项目"""
        project = self.get_project(project_id)
        if not project:
            return False

        # 删除项目数据目录
        project_name = project["name"]
        project_dir = get_root_dir() / "data" / project_name
        if project_dir.exists():
            try:
                shutil.rmtree(project_dir)
            except Exception as e:
                print(f"删除项目目录失败: {e}")

        # 从配置中移除
        self.config["projects"] = [
            p for p in self.config["projects"]
            if p["id"] != project_id
        ]

        # 如果删除的是当前项目，清除当前项目ID
        if self.config.get("current_project_id") == project_id:
            self.config["current_project_id"] = None

        self.save_config()
        return True

    def set_current_project(self, project_id: str) -> bool:
        """设置当前项目"""
        project = self.get_project(project_id)
        if not project:
            return False

        self.config["current_project_id"] = project_id
        # 更新最后使用时间
        self.update_project(project_id, {"last_used": format_timestamp()})
        return True

    def get_current_project(self) -> Optional[Dict]:
        """获取当前选中的项目（fallback 优先选最近一个 ready 的项目）

        优先级：
        1. current_project_id 指向的项目为 ready → 返回它
        2. current_project_id 指向的项目不存在 / 不是 ready → fallback 到最近 ready
        3. 没 current_project_id → fallback 到最近 ready
        4. 都没有 ready 项目 → 兜底返回最后一个项目
        """
        current_id = self.config.get("current_project_id")
        if current_id:
            p = self.get_project(current_id)
            if p and p.get("status") == "ready":
                return p
        # Fallback: 最近一个 ready 项目
        projects = self.config.get("projects", [])
        ready_projects = [p for p in projects if p.get("status") == "ready"]
        if ready_projects:
            return ready_projects[-1]
        # 最后兑底
        if projects:
            return projects[-1]
        return None

    def clear_current_project(self):
        """清除当前项目"""
        self.config["current_project_id"] = None
        self.save_config()

    def get_project_count(self) -> int:
        """获取项目数量"""
        return len(self.config.get("projects", []))

    def project_name_exists(self, name: str) -> bool:
        """检查项目名称是否已存在"""
        return any(p["name"] == name for p in self.config.get("projects", []))

    def get_project_status(self, project_id: str) -> str:
        """获取项目状态"""
        project = self.get_project(project_id)
        if not project:
            return "unknown"

        vector_db_path = project.get("vector_db_path", "")
        if not os.path.exists(vector_db_path):
            return "not_ready"

        # 检查向量化文件名（支持自定义命名）
        vector_file = project.get("vector_file", "")
        metadata_file = project.get("metadata_file", "")

        if vector_file and metadata_file:
            # 使用指定的文件名
            has_embeddings = os.path.exists(os.path.join(vector_db_path, vector_file))
            has_metadata = os.path.exists(os.path.join(vector_db_path, metadata_file))
        else:
            # 使用默认文件名
            has_embeddings = os.path.exists(os.path.join(vector_db_path, "embeddings.npy"))
            has_metadata = os.path.exists(os.path.join(vector_db_path, "metadata.json"))

        # 如果默认文件不存在，尝试查找目录中的npy文件
        if not has_embeddings:
            npy_files = [f for f in os.listdir(vector_db_path) if f.endswith('.npy')]
            if npy_files:
                # 查找对应的json文件
                for npy_file in npy_files:
                    json_file = npy_file.replace('.npy', '.json')
                    if os.path.exists(os.path.join(vector_db_path, json_file)):
                        has_embeddings = True
                        has_metadata = True
                        # 自动更新配置
                        self.update_project(project_id, {
                            "vector_file": npy_file,
                            "metadata_file": json_file
                        })
                        break

        if has_embeddings and has_metadata:
            return "ready"
        # 仅有 metadata.json（无向量）→ 关键词检索模式，项目可用但召回质量差
        if has_metadata:
            return "keyword_only"
        if os.path.exists(project.get("cleaned_path", "")):
            return "cleaned"
        elif os.path.exists(project.get("source_path", "")):
            return "created"
        else:
            return "error"

    def get_project_runtime(self, project_id: str) -> Dict:
        """获取项目运行时信息（所见即所得的真相来源）

        返回字段：
        - status: 同 get_project_status
        - vector_db_path: 实际向量库路径
        - is_external: vector_db_path 不在项目标准 data/<name>/ 下
        - chunk_count: metadata.json 实际段数（0 表示不可读）
        - has_embeddings: embeddings.npy 存在
        - has_metadata: metadata.json 存在
        - embeddings_size_mb: embeddings.npy 大小（MB）
        - source_size_mb: 原文件大小（MB）
        - error: 异常信息
        """
        project = self.get_project(project_id)
        if not project:
            return {"status": "unknown", "error": "项目不存在"}

        vector_db_path = project.get("vector_db_path", "")
        source_path = project.get("source_path", "")
        result = {
            "status": self.get_project_status(project_id),
            "vector_db_path": vector_db_path,
            "is_external": False,
            "chunk_count": 0,
            "has_embeddings": False,
            "has_metadata": False,
            "embeddings_size_mb": 0.0,
            "source_size_mb": 0.0,
        }

        # 判断 vector_db_path 是否在项目标准 data/<name>/ 下
        if vector_db_path:
            project_data_dir = str(get_data_dir() / project.get("name", ""))
            try:
                rel = os.path.relpath(vector_db_path, project_data_dir)
                result["is_external"] = rel.startswith("..") or os.path.isabs(rel)
            except (ValueError, OSError):
                result["is_external"] = True

        # 查文件
        vector_file = project.get("vector_file", "")
        metadata_file = project.get("metadata_file", "")
        if not vector_file:
            vector_file = "embeddings.npy"
        if not metadata_file:
            metadata_file = "metadata.json"
        npy_path = os.path.join(vector_db_path, vector_file) if vector_db_path else ""
        meta_path = os.path.join(vector_db_path, metadata_file) if vector_db_path else ""

        try:
            if npy_path and os.path.isfile(npy_path):
                result["has_embeddings"] = True
                result["embeddings_size_mb"] = round(os.path.getsize(npy_path) / 1024 / 1024, 1)
            if meta_path and os.path.isfile(meta_path):
                result["has_metadata"] = True
                # 读 metadata 拿段数（仅 0 段/小文件快，大文件不读全）
                import json
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    if isinstance(meta, list):
                        result["chunk_count"] = len(meta)
                except (json.JSONDecodeError, OSError):
                    pass
        except OSError as e:
            result["error"] = str(e)

        if source_path and os.path.isfile(source_path):
            result["source_size_mb"] = round(os.path.getsize(source_path) / 1024 / 1024, 1)

        return result

    def is_keyword_only(self, project_id: str) -> bool:
        """检查项目是否处于纯关键词检索模式（metadata 在但 embeddings.npy 缺失）"""
        return self.get_project_status(project_id) == "keyword_only"

    def import_existing_vector(
        self,
        name: str,
        source_path: str,
        vector_db_path: str
    ) -> Optional[Dict]:
        """
        导入已有的向量化文件，创建新项目

        Args:
            name: 小说名称
            source_path: 原始小说文件路径（可选）
            vector_db_path: 向量化文件所在目录路径

        Returns:
            新项目配置字典，如果失败则返回None
        """
        if not os.path.exists(vector_db_path):
            print(f"错误: 向量化目录不存在: {vector_db_path}")
            return None

        # 查找向量化文件
        npy_files = [f for f in os.listdir(vector_db_path) if f.endswith('.npy')]
        if not npy_files:
            print(f"错误: 目录中未找到.npy文件: {vector_db_path}")
            return None

        # 查找匹配的json文件（兼容多种命名习惯：同名的 <stem>.json、
        # 项目标准 metadata.json、以及常见别名）
        vector_file = None
        metadata_file = None

        def _candidate_jsons_for(npy):
            stem = os.path.splitext(npy)[0]
            candidates = [
                npy.replace('.npy', '.json'),  # embeddings.npy -> embeddings.json
                'metadata.json',                # 项目标准
                'embeddings.json',              # 别名
                f'{stem}.meta.json',            # embeddings.meta.json
            ]
            seen = set()
            result = []
            for cand in candidates:
                if cand not in seen:
                    seen.add(cand)
                    result.append(cand)
            return result

        for npy_file in npy_files:
            for json_file in _candidate_jsons_for(npy_file):
                if os.path.exists(os.path.join(vector_db_path, json_file)):
                    vector_file = npy_file
                    metadata_file = json_file
                    break
            if vector_file:
                break

        if not vector_file:
            print(f"错误: 未找到匹配的.json元数据文件")
            return None

        # 创建项目目录
        project_dir = get_project_dir(name)
        dest_vector_db = project_dir / "vector_db"

        # 复制向量化文件到项目目录
        dest_vector_db.mkdir(exist_ok=True)
        source_npy = os.path.join(vector_db_path, vector_file)
        source_json = os.path.join(vector_db_path, metadata_file)
        dest_npy = os.path.join(str(dest_vector_db), vector_file)
        dest_json = os.path.join(str(dest_vector_db), metadata_file)

        # 如果目标文件已存在，先删除
        if os.path.exists(dest_npy):
            os.remove(dest_npy)
        if os.path.exists(dest_json):
            os.remove(dest_json)

        shutil.copy2(source_npy, dest_npy)
        shutil.copy2(source_json, dest_json)

        # 创建项目配置
        project_id = generate_project_id()
        source_dir = project_dir / "source"
        source_dir.mkdir(exist_ok=True)

        # 如果有源文件，也复制过去
        if source_path and os.path.exists(source_path):
            source_dest = source_dir / os.path.basename(source_path)
            if not source_dest.exists():
                shutil.copy2(source_path, str(source_dest))

        project_config = {
            "id": project_id,
            "name": name,
            "source_path": str(source_dir / os.path.basename(source_path)) if source_path else "",
            "cleaned_path": "",
            "vector_db_path": str(dest_vector_db),
            "vector_file": vector_file,
            "metadata_file": metadata_file,
            "chunk_size": 672,
            "overlap": 50,
            "clean_rules": [],
            "custom_dirty_words": [],
            "created_at": format_timestamp(),
            "last_used": format_timestamp(),
            "status": "ready"
        }

        self.config["projects"].append(project_config)
        self.save_config()

        print(f"✓ 成功导入向量化文件: {vector_file}")
        print(f"  → 项目「{name}」已创建")
        return project_config
