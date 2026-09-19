# -*- coding: utf-8 -*-
"""
tests/test_embedding_local_first.py — 嵌入模型「本地优先」加载链路单元测试

验证目标：指定的嵌入模型（如 BAAI/bge-small-zh-v1.5）
  1) 先查项目根 models/ 目录，命中则直接用本地路径离线加载（local_files_only=True），不联网；
  2) 仅当本地不存在时才联网下载到 models/ 后再加载；
  3) 下载失败时兜底 HF 默认缓存，返回 (None, cpu) 而不抛异常；
  4) 空壳目录（models/<name>/ 无权重，下载中断残留）不会锁死本地优先，改走重新下载；
  5) 下载实现兼容 huggingface_hub 0.x / 1.x（1.x 已移除 local_dir_use_symlinks），
     且失败后不残留空壳目录。

全部用例均离线执行：所有网络相关调用都被 monkeypatch 替换。
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils

MODEL_ID = "BAAI/bge-small-zh-v1.5"


def _make_model_dir(path: Path, weights: str | None = "model.safetensors") -> Path:
    """构造「可离线加载」的最小模型目录：config.json + 权重文件（weights=None 则只留 config）"""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (path / weights).write_bytes(b"\x00" * 16)
    return path


class _FakeSentenceTransformer:
    """记录构造参数的 SentenceTransformer 替身（支持实例属性赋值）"""

    calls: list = []

    def __init__(self, *args, **kwargs):
        _FakeSentenceTransformer.calls.append({"args": args, "kwargs": kwargs})
        self.max_seq_length = 0
        self.device = kwargs.get("device", "cpu")


class _FakeCrossEncoder:
    """记录构造参数的 CrossEncoder 替身"""

    calls: list = []

    def __init__(self, *args, **kwargs):
        _FakeCrossEncoder.calls.append({"args": args, "kwargs": kwargs})
        self.max_seq_length = 0
        self.device = kwargs.get("device", "cpu")


@pytest.fixture
def fake_st(monkeypatch):
    pytest.importorskip("sentence_transformers")
    import sentence_transformers as st
    _FakeSentenceTransformer.calls = []
    monkeypatch.setattr(st, "SentenceTransformer", _FakeSentenceTransformer)
    return _FakeSentenceTransformer


@pytest.fixture
def fake_ce(monkeypatch):
    pytest.importorskip("sentence_transformers")
    import sentence_transformers as st
    _FakeCrossEncoder.calls = []
    monkeypatch.setattr(st, "CrossEncoder", _FakeCrossEncoder)
    return _FakeCrossEncoder


@pytest.fixture
def project(tmp_path, monkeypatch):
    """伪项目根（含空 models/），设备固定为 CPU，避免真实 GPU 探测"""
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(utils, "get_root_dir", lambda: tmp_path)
    monkeypatch.setattr(utils, "_GPU_DEVICE", "cpu")
    monkeypatch.setattr(utils, "_GPU_AVAILABLE", False)
    return tmp_path


@pytest.fixture
def no_download(monkeypatch):
    """本地命中时若触发下载则直接失败"""
    def _boom(model_id):
        raise AssertionError(f"本地命中却触发了联网下载：{model_id}")
    monkeypatch.setattr(utils, "_try_download_to_local_models", _boom)


class TestLocalHitLoadsOffline:
    """models/ 命中 → 本地路径 + local_files_only=True 离线加载"""

    def test_hf_id_hit_uses_local_path_offline(self, project, fake_st, no_download):
        local = _make_model_dir(project / "models" / "bge-small-zh-v1.5")
        model, device = utils.load_embedding_model(MODEL_ID)
        assert model is not None
        assert device == "cpu"
        call = fake_st.calls[0]
        assert Path(str(call["args"][0])).resolve() == local.resolve()
        assert call["kwargs"]["local_files_only"] is True

    def test_relative_path_hit_uses_local_path_offline(self, project, fake_st, no_download):
        """相对路径（GUI 启动时 cwd 不在项目根）也应按项目根解析到本地"""
        local = _make_model_dir(project / "models" / "bge-local-probe")
        model, _ = utils.load_embedding_model("models/bge-local-probe")
        assert model is not None
        call = fake_st.calls[0]
        assert Path(str(call["args"][0])).resolve() == local.resolve()
        assert call["kwargs"]["local_files_only"] is True

    def test_reranker_local_hit_loads_from_local_path(self, project, fake_ce, no_download):
        local = _make_model_dir(project / "models" / "bge-reranker-base")
        model, _ = utils.load_reranker_model("BAAI/bge-reranker-base")
        assert model is not None
        assert Path(str(fake_ce.calls[0]["args"][0])).resolve() == local.resolve()


class TestLocalMissDownloadsThenLoads:
    """本地未命中 → 先下载到 models/，再离线加载"""

    def test_missing_local_dir_downloads_to_models_then_offline_load(self, project, fake_st, monkeypatch):
        seen = []

        def fake_download(model_id):
            seen.append(model_id)
            return str(_make_model_dir(project / "models" / model_id.split("/")[-1]))

        monkeypatch.setattr(utils, "_try_download_to_local_models", fake_download)
        model, _ = utils.load_embedding_model(MODEL_ID)
        assert seen == [MODEL_ID]
        call = fake_st.calls[0]
        assert Path(str(call["args"][0])).resolve() == (project / "models" / "bge-small-zh-v1.5").resolve()
        assert call["kwargs"]["local_files_only"] is True
        assert model is not None

    def test_download_failure_falls_back_to_hub_cache(self, project, fake_st, monkeypatch):
        monkeypatch.setattr(utils, "_try_download_to_local_models", lambda model_id: None)
        utils.load_embedding_model(MODEL_ID)
        call = fake_st.calls[0]
        assert call["args"][0] == MODEL_ID
        assert "local_files_only" not in call["kwargs"]

    def test_empty_shell_dir_does_not_lock_local_first(self, project, fake_st, monkeypatch):
        """空壳目录（下载中断残留）→ 应改判为未命中并重新下载，而非永久离线加载失败"""
        (project / "models" / "bge-small-zh-v1.5").mkdir(parents=True)
        seen = []

        def fake_download(model_id):
            seen.append(model_id)
            return str(_make_model_dir(project / "models" / model_id.split("/")[-1]))

        monkeypatch.setattr(utils, "_try_download_to_local_models", fake_download)
        model, _ = utils.load_embedding_model(MODEL_ID)
        assert seen == [MODEL_ID]
        assert model is not None
        assert fake_st.calls[0]["kwargs"]["local_files_only"] is True

    def test_incomplete_local_dir_without_weights_reroutes_to_download(self, project, fake_st, monkeypatch):
        """只有 config.json、没有权重的半截目录同样不算命中"""
        _make_model_dir(project / "models" / "bge-small-zh-v1.5", weights=None)
        seen = []
        monkeypatch.setattr(
            utils,
            "_try_download_to_local_models",
            lambda model_id: seen.append(model_id) or str(
                _make_model_dir(project / "models" / model_id.split("/")[-1])
            ),
        )
        model, _ = utils.load_embedding_model(MODEL_ID)
        assert seen == [MODEL_ID]
        assert model is not None

    def test_explicit_local_path_keeps_legacy_behavior(self, project, fake_st, no_download):
        """显式传本地路径时不改判（目录有效性由调用方负责），保持历史行为"""
        local = _make_model_dir(project / "models" / "custom-model", weights=None)
        utils.load_embedding_model(str(local))
        assert Path(str(fake_st.calls[0]["args"][0])).resolve() == local.resolve()


class TestDownloadHelperCompat:
    """_try_download_to_local_models：hub 版本兼容 + 失败不留空壳"""

    def test_modern_hub_signature_no_typeerror(self, project, monkeypatch):
        captured = {}

        class FakeHub:
            @staticmethod
            def snapshot_download(repo_id, *, local_dir=None, cache_dir=None, ignore_patterns=None, **kwargs):
                if "local_dir_use_symlinks" in kwargs:
                    raise TypeError(
                        "snapshot_download() got an unexpected keyword argument 'local_dir_use_symlinks'"
                    )
                captured["repo_id"] = repo_id
                captured["cache_dir"] = cache_dir
                captured["ignore_patterns"] = ignore_patterns
                _make_model_dir(Path(local_dir))
                return str(local_dir)

        monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
        result = utils._try_download_to_local_models(MODEL_ID)
        assert result == str(project / "models" / "bge-small-zh-v1.5")
        assert captured["repo_id"] == MODEL_ID
        assert captured["cache_dir"] == str(project / ".hf_cache")
        # vocab.txt 等 tokenizer 依赖的文本资源不能被忽略
        assert "*.txt" not in (captured["ignore_patterns"] or [])

    def test_legacy_hub_receives_symlink_flag(self, project, monkeypatch):
        captured = {}

        class FakeHub:
            @staticmethod
            def snapshot_download(repo_id, local_dir=None, local_dir_use_symlinks=True,
                                  cache_dir=None, ignore_patterns=None):
                captured["local_dir_use_symlinks"] = local_dir_use_symlinks
                _make_model_dir(Path(local_dir))
                return str(local_dir)

        monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
        assert utils._try_download_to_local_models(MODEL_ID) is not None
        assert captured["local_dir_use_symlinks"] is False

    def test_failure_leaves_no_empty_shell_dir(self, project, monkeypatch):
        class FakeHub:
            @staticmethod
            def snapshot_download(repo_id, *, local_dir=None, **kwargs):
                Path(local_dir).mkdir(parents=True, exist_ok=True)  # 新版 hub 会先建目录
                raise RuntimeError("network down")

        monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
        assert utils._try_download_to_local_models(MODEL_ID) is None
        assert not (project / "models" / "bge-small-zh-v1.5").exists()

    def test_incomplete_download_returns_none(self, project, monkeypatch):
        class FakeHub:
            @staticmethod
            def snapshot_download(repo_id, *, local_dir=None, **kwargs):
                d = Path(local_dir)
                d.mkdir(parents=True, exist_ok=True)
                (d / "tokenizer.json").write_text("{}", encoding="utf-8")
                return str(d)

        monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
        assert utils._try_download_to_local_models(MODEL_ID) is None


class TestReadinessProbe:
    """is_local_model_ready / is_hf_model_id 判定"""

    def test_empty_dir_not_ready(self, tmp_path):
        assert utils.is_local_model_ready(tmp_path) is False

    def test_config_only_not_ready(self, tmp_path):
        _make_model_dir(tmp_path, weights=None)
        assert utils.is_local_model_ready(tmp_path) is False

    def test_config_plus_weights_ready(self, tmp_path):
        _make_model_dir(tmp_path)
        assert utils.is_local_model_ready(tmp_path) is True

    def test_onnx_only_ready(self, tmp_path):
        _make_model_dir(tmp_path, weights=None)
        (tmp_path / "onnx").mkdir()
        assert utils.is_local_model_ready(tmp_path) is True

    def test_missing_dir_not_ready(self, tmp_path):
        assert utils.is_local_model_ready(tmp_path / "no-such-model") is False

    def test_is_hf_model_id(self):
        assert utils.is_hf_model_id("BAAI/bge-small-zh-v1.5") is True
        assert utils.is_hf_model_id("bge-small-zh-v1.5") is False
        assert utils.is_hf_model_id("") is False
        assert utils.is_hf_model_id(None) is False


class TestPublicApiCompatibility:
    """函数签名保持不变，调用方无需改动"""

    def test_single_positional_arg_signatures(self):
        cases = (
            (utils.resolve_embedding_model, "spec"),
            (utils.load_embedding_model, "spec"),
            (utils.load_reranker_model, "spec"),
            (utils._try_download_to_local_models, "model_id"),
        )
        for fn, param_name in cases:
            params = list(inspect.signature(fn).parameters.values())
            assert len(params) == 1, f"{fn.__name__} 参数数量变了：{params}"
            assert params[0].name == param_name
            assert params[0].default is inspect.Parameter.empty
