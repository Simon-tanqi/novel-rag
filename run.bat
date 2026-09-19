@echo off
REM 一键启动 novel-rag GUI。
REM 必须使用项目虚拟环境解释器：系统 Python 未安装 torch/sentence-transformers，
REM 用其启动会导致嵌入模型加载失败 → 建库不生成 embeddings.npy → 向量检索静默降级为关键词模式。
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    echo [错误] 未找到 .venv 虚拟环境，请先运行 setup_env.bat
    pause
)
