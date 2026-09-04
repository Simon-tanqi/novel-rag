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
      "chunk_size": 500,
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

    def __init__(self):
        self.config_file = get_root_dir() / "novel_config.json"
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
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
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"保存项目配置失败: {e}")
            raise

    # ===================== 项目CRUD =====================

    def create_project(
        self,
        name: str,
        source_path: str,
        chunk_size: int = 500,
        overlap: int = 50,
        clean_rules: Optional[List[str]] = None,
        custom_dirty_words: Optional[List[str]] = None
    ) -> Dict:
        """
        创建新项目

        Args:
            name: 小说名称
            source_path: 源文件路径
            chunk_size: 切片大小
            overlap: 重叠长度
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
        """获取当前选中的项目"""
        current_id = self.config.get("current_project_id")
        if current_id:
            return self.get_project(current_id)
        # 如果没有当前项目，返回最后一个项目（如果存在）
        projects = self.config.get("projects", [])
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
        elif os.path.exists(project.get("cleaned_path", "")):
            return "cleaned"
        elif os.path.exists(project.get("source_path", "")):
            return "created"
        else:
            return "error"

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

        # 查找匹配的json文件
        vector_file = None
        metadata_file = None
        for npy_file in npy_files:
            json_file = npy_file.replace('.npy', '.json')
            if os.path.exists(os.path.join(vector_db_path, json_file)):
                vector_file = npy_file
                metadata_file = json_file
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
            "chunk_size": 500,
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
