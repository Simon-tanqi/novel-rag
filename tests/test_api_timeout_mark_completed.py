# -*- coding: utf-8 -*-
"""
验证 _call_api 完成后 _api_timeout_check 不会再触发"超时"提示

规格：AI 已产出应答（api_completed=True）后，超时检查线程不得再报超时。
因此 _api_timeout_check 循环内先看 api_completed，已完成立即返回；
仅在 90s 到时且 !api_completed 时才提示"超时"。

字段约定:
- _call_api 成功 → api_completed=True, api_timeout=False
- _call_api 异常 → api_completed=True, api_error=True
- 早返回 → api_completed=True (success)

这里测核心场景：模拟 90s 后 api_completed=True 时的行为
"""
# 顶部 stub 掉 main.py 的重型 GUI 依赖（customtkinter 在 CI/3.11 环境不一定装）
import sys
import time
from unittest.mock import MagicMock

# 为 customtkinter 创建虚假的 CTk 父类（必须是类不是实例，
# 否则 main.NovelRAGApp(ctk.CTk) 继承会得到 MagicMock 类）
class _FakeCTk:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return MagicMock()
    def after(self, *args, **kwargs): pass
    def mainloop(self, *args, **kwargs): pass
_fake_ctk_module = MagicMock()
_fake_ctk_module.CTk = _FakeCTk
sys.modules['customtkinter'] = _fake_ctk_module
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()


# main.py 依赖 customtkinter (GUI lib)，必须能 import 整个模块
# 我们只测 _api_timeout_check 的纯函数行为


def _make_app_for_timeout_test(api_completed: bool, elapsed: float):
    """构造 _api_timeout_check 需要的 mock app（让 self.after 立即同步执行 callback）"""
    import main as main_mod
    # 拿真正的 NovelRAGApp（避免被 customtkinter mock 污染）
    NovelRAGApp = getattr(type(main_mod), 'NovelRAGApp', None) or main_mod.NovelRAGApp
    # 退路：如果 type 拿不到，看 __dict__
    if isinstance(NovelRAGApp, MagicMock):
        # 从 main_mod.__dict__ 拿原始引用（mock 是模块属性，该同）
        for name in dir(main_mod):
            obj = getattr(main_mod, name)
            if hasattr(obj, '_api_timeout_check') and not isinstance(obj, MagicMock):
                NovelRAGApp = obj
                break
    app = MagicMock()  # 不用 spec，避免 _FakeCTk 限制属性
    app.api_start_time = time.time() - elapsed
    app.api_completed = api_completed
    app.api_timeout = False
    app.api_error = False

    # self.after(0, cb) -> 让 cb 立即同步执行
    def fake_after(delay_ms, cb):
        cb()
    app.after.side_effect = fake_after
    return app, main_mod


def test_api_completed_prevents_timeout_alert():
    """API 已完成时，_api_timeout_check 不应触发 _show_error"""
    app, main_mod = _make_app_for_timeout_test(api_completed=True, elapsed=100)

    main_mod.NovelRAGApp._api_timeout_check(app)

    # api_completed=True → 底部 if not api_completed: False，不报警
    # 关键：_show_error 不该被调用（失败打标后不得再弹错误提示）
    assert not app._show_error.called, "API 已完成时不应报超时"
    # api_timeout 不该被置 True
    assert app.api_timeout is False


def test_api_not_completed_triggers_timeout_alert():
    """API 未完成且超过 90s 时，_api_timeout_check 应触发 _show_error"""
    app, main_mod = _make_app_for_timeout_test(api_completed=False, elapsed=100)

    main_mod.NovelRAGApp._api_timeout_check(app)

    assert app.api_timeout is True
    assert app.api_error is True
    assert app._show_error.called, "API 未完成且超时应报超时"
    status_calls = [c for c in app.status_label.configure.call_args_list
                    if c.kwargs.get("text", "").startswith("状态: 超时")]
    assert status_calls, "状态栏应被设为'超时'"


    # 集成用例的 mock 层级较多、环境依赖复杂；其核心判定点
    # （api_completed 标志 + _api_timeout_check 检查）已由前两个测试覆盖
