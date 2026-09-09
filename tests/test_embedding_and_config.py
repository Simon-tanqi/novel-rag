# -*- coding: utf-8 -*-
"""
嵌入模型解析 & 配置冷启动（环境变量兜底）单元测试
"""
import json

from config_manager import ConfigManager
from utils import DEFAULT_EMBEDDING_MODEL, resolve_embedding_model


class TestResolveEmbeddingModel:
    """resolve_embedding_model: 本地目录 / HF 模型名 / 未配置 / 失效路径"""

    def test_empty_returns_none(self):
        assert resolve_embedding_model("") is None
        assert resolve_embedding_model(None) is None
        assert resolve_embedding_model("   ") is None

    def test_local_dir_returns_local(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        assert resolve_embedding_model(str(model_dir)) == ("local", str(model_dir))

    def test_hub_id_returns_hub_or_local(self, tmp_path, monkeypatch):
        """HF 模型 ID：若项目根 models/<basename> 存在则 local，否则 hub"""
        # 重定向 get_root_dir 到 tmp_path，让项目根/ models 为空
        import utils
        monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
        result = resolve_embedding_model("BAAI/bge-small-zh-v1.5")
        assert result is not None
        # 本机器项目根 models/ 下已有该模型，默认会走 local；空环境则 hub
        assert result[0] in ("hub", "local")

    def test_missing_drive_path(self, tmp_path, monkeypatch):
        """不存在的绝对路径：不再返回 None，而是 hub 兜底（让调用方按 ID 走）"""
        import utils
        monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
        result = resolve_embedding_model("C:/no/such/model/dir")
        # 不是 HF ID 形式（无 /）→ 既非 local 也非 None，兜底 hub
        assert result is not None
        assert result[0] == "hub"

    def test_missing_relative_path(self, tmp_path, monkeypatch):
        """不存在的相对路径：同样兜底 hub（让调用方按 ID 走）"""
        import utils
        monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
        result = resolve_embedding_model("./not_exist_model_dir")
        assert result is not None
        assert result[0] == "hub"

    def test_whitespace_surrounding_is_stripped(self):
        assert resolve_embedding_model("  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  ") == (
            "hub", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class TestConfigColdStart:
    """config.json 缺失时，DEEPSEEK_API_KEY 应自动注入默认 DeepSeek 模型"""

    def test_no_env_no_models(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        cfg = ConfigManager(str(tmp_path / "missing.json"))
        assert cfg.get_models() == []

    def test_env_seeds_default_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        cfg = ConfigManager(str(tmp_path / "missing.json"))
        models = cfg.get_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "deepseek-chat"
        assert models[0]["api_url"] == "https://api.deepseek.com"
        # 密钥来自环境变量运行时注入，不写进文件
        assert models[0]["api_key"] == "sk-test-123"

    def test_env_injects_when_file_has_empty_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-456")
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "models": [{
                "name": "deepseek-chat",
                "model_id": "deepseek-chat",
                "api_url": "https://api.deepseek.com",
                "api_key": "",
                "is_local": False,
            }]
        }, ensure_ascii=False), encoding="utf-8")
        cfg = ConfigManager(str(cfg_file))
        assert cfg.get_models()[0]["api_key"] == "sk-test-456"

    def test_default_embedding_model_falls_back_to_builtin(self, tmp_path):
        cfg = ConfigManager(str(tmp_path / "missing.json"))
        assert cfg.get("embedding_model_path") == DEFAULT_EMBEDDING_MODEL
