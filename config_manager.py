"""
config_manager.py — 配置管理器
管理全局配置 config.json（模型、API、Prompt等）
"""
import json
import os
from pathlib import Path
from typing import Any, Optional


class ConfigManager:
    """全局配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_file: 配置文件路径，默认为项目根目录下的 config.json
        """
        if config_file is None:
            from utils import get_root_dir
            self.config_file = str(get_root_dir() / "config.json")
        else:
            self.config_file = config_file

        self.config = self.load()

    def load(self) -> dict:
        """
        加载配置文件

        Returns:
            配置字典
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠ 加载配置文件失败: {e}")
                return self._get_default_config()
        return self._get_default_config()

    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            "models": [],
            "prompt_template": self._get_default_prompt(),
            "top_k": 3,
            "enable_thinking": False,
            "enable_rerank": False,
            "reranker_model_path": "",
            "embedding_model_path": "",
            "vector_folder": "",
            "local_model_path": ""
        }

    def _get_default_prompt(self) -> str:
        """获取默认Prompt模板"""
        return (
            "你是一个专门根据提供的《{novel_name}》小说原文片段回答问题的助手，精通《{novel_name}》的世界观、人物、剧情和细节。\n\n"
            "重要规则：\n"
            "1. 只使用提供的《{novel_name}》原文片段中的信息回答问题\n"
            "2. 如果原文片段中没有相关信息，就说\"根据提供的《{novel_name}》原文片段无法回答此问题\"\n"
            "3. 如果有相关信息，直接引用原文片段的内容\n"
            "4. 不要编造《{novel_name}》原文中没有的信息\n"
            "5. 不要回答与《{novel_name}》原文无关的内容\n"
            "6. 回答时请明确指出内容出自《{novel_name}》\n\n"
            "以下是提供的《{novel_name}》原文片段：\n"
            "{context}\n\n"
            "用户关于《{novel_name}》的问题：{question}\n\n"
            "根据《{novel_name}》原文片段回答："
        )

    def save(self):
        """保存配置文件"""
        try:
            # 确保目录存在
            config_dir = os.path.dirname(self.config_file)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)

        except IOError as e:
            print(f"✗ 保存配置失败: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """
        设置配置值（不自动保存）

        Args:
            key: 配置键
            value: 配置值
        """
        self.config[key] = value

    def get_models(self) -> list:
        """
        获取所有模型列表

        API Key 优先级：
        1. 配置文件中的 api_key
        2. 环境变量 DEEPSEEK_API_KEY（当配置文件为空时自动兜底，
           避免把密钥写进仓库/配置）
        """
        models = self.config.get("models", [])
        env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if env_key:
            resolved = []
            for model in models:
                model = dict(model)
                if not model.get("api_key"):
                    model["api_key"] = env_key
                resolved.append(model)
            return resolved
        return models

    def get_active_model(self) -> Optional[dict]:
        """获取当前选中的模型"""
        models = self.get_models()
        if models:
            return models[0]
        return None

    def add_model(self, model_config: dict) -> bool:
        """
        添加新模型

        Args:
            model_config: 模型配置字典

        Returns:
            是否成功
        """
        models = self.get_models()

        # 检查是否已存在同名模型
        for model in models:
            if model["name"] == model_config.get("name"):
                return False

        models.append(model_config)
        self.config["models"] = models
        self.save()
        return True

    def update_model(self, model_name: str, model_config: dict) -> bool:
        """
        更新模型配置

        Args:
            model_name: 模型名称
            model_config: 新配置

        Returns:
            是否成功
        """
        models = self.get_models()
        for i, model in enumerate(models):
            if model["name"] == model_name:
                models[i] = model_config
                self.config["models"] = models
                self.save()
                return True
        return False

    def delete_model(self, model_name: str) -> bool:
        """
        删除模型

        Args:
            model_name: 模型名称

        Returns:
            是否成功
        """
        models = self.get_models()
        new_models = [m for m in models if m["name"] != model_name]
        if len(new_models) == len(models):
            return False
        self.config["models"] = new_models
        self.save()
        return True

    def get_prompt_template(self) -> str:
        """获取Prompt模板"""
        return self.get("prompt_template", self._get_default_prompt())

    def set_prompt_template(self, template: str):
        """设置Prompt模板"""
        self.set("prompt_template", template)
        self.save()

    def get_embedding_model_path(self) -> str:
        """获取嵌入模型路径"""
        return self.get("embedding_model_path", "")

    def set_embedding_model_path(self, path: str):
        """设置嵌入模型路径"""
        self.set("embedding_model_path", path)
        self.save()

    def get_reranker_model_path(self) -> str:
        """获取重排模型路径"""
        return self.get("reranker_model_path", "")

    def set_reranker_model_path(self, path: str):
        """设置重排模型路径"""
        self.set("reranker_model_path", path)
        self.save()
