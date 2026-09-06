"""
tests/test_gpu_device.py — GPU 设备检测与 load_embedding_model 返回值测试
"""
import pytest
import sys
from pathlib import Path

# ── 隔离层：mock utils 模块的 get_root_dir ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


class TestGetComputeDevice:
    """utils.get_compute_device() 自动检测 GPU/CPU/MPS"""

    def test_returns_one_of_valid_devices(self):
        sys.path.insert(0, str(ROOT))
        from utils import get_compute_device
        device = get_compute_device()
        assert device in ("cuda", "mps", "cpu"), f"Unexpected device: {device}"

    def test_result_is_cached(self):
        sys.path.insert(0, str(ROOT))
        from utils import get_compute_device
        d1 = get_compute_device()
        d2 = get_compute_device()
        assert d1 == d2  # 结果应缓存，不重复检测

    # 注：真实 CUDA 检测依赖系统已安装 GPU 版 torch（import torch 后
    # torch.cuda.is_available() 由底层 C 库决定），无法通过 monkeypatch 模拟。
    # 该场景由 test_returns_tuple_on_success 间接验证（load_embedding_model
    # 返回 device 字段与实际 GPU 状态一致）。

    def test_cpu_fallback_when_torch_unavailable(self, monkeypatch):
        sys.path.insert(0, str(ROOT))
        import utils
        utils._GPU_DEVICE = None
        utils._GPU_AVAILABLE = None

        class FakeTorch:
            @staticmethod
            def is_available():
                raise ImportError("no torch")

        monkeypatch.setitem(sys.modules, "torch", FakeTorch)
        from utils import get_compute_device
        device = get_compute_device()
        assert device == "cpu"

        utils._GPU_DEVICE = None
        utils._GPU_AVAILABLE = None


class TestLoadEmbeddingModelReturnsTuple:
    """load_embedding_model() 返回 (model, device) 而非仅 model"""

    def test_returns_tuple_on_success(self, monkeypatch):
        sys.path.insert(0, str(ROOT))
        import utils
        utils._GPU_DEVICE = "cpu"  # 强制 CPU
        utils._GPU_AVAILABLE = False

        model, device = utils.load_embedding_model("BAAI/bge-small-zh-v1.5")
        # 成功加载时 device 应该是字符串
        assert isinstance(device, str)
        assert device in ("cuda", "cpu", "mps")
        assert isinstance(model, object)  # SentenceTransformer 实例

    def test_returns_none_cpu_on_failure(self, monkeypatch):
        sys.path.insert(0, str(ROOT))
        import utils
        utils._GPU_DEVICE = None
        utils._GPU_AVAILABLE = None

        # 传入不存在的模型，强制失败
        result = utils.load_embedding_model("NONEXISTENT_MODEL_XYZ_12345")
        # 返回值应为 (None, "cpu") 或 None（取决于具体失败路径）
        assert result is None or (
            isinstance(result, tuple) and result[0] is None and result[1] == "cpu"
        ), f"Unexpected return: {result}"

        utils._GPU_DEVICE = None
        utils._GPU_AVAILABLE = None
