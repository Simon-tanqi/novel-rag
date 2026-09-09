@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  setup_env.bat - 一键创建 Python 3.13 虚拟环境并安装依赖
REM  适用：Windows + 已安装 Python 3.13
REM  用法：双击运行，或在 PowerShell/Cmd 中执行 .\setup_env.bat
REM ============================================================

echo.
echo ============================================================
echo   Novel-RAG 环境一键设置
echo ============================================================
echo.

REM 1) 检查 py launcher
where py >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 py launcher
    echo 请先安装 Python 3.13：https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 2) 检查 Python 3.13 是否存在
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未安装 Python 3.13
    echo 请从 https://www.python.org/downloads/ 下载 3.13 并勾选"Add Python to PATH"
    pause
    exit /b 1
)

echo [1/4] 检测到 Python 3.13：
py -3.13 --version
echo.

REM 3) 创建 .venv
if not exist .venv (
    echo [2/4] 创建虚拟环境 .venv ...
    py -3.13 -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
) else (
    echo [2/4] 虚拟环境 .venv 已存在，跳过创建
)
echo.

REM 4) 激活 + 升级 pip + 装依赖
echo [3/4] 激活虚拟环境并安装依赖 ...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [警告] pip 升级失败，继续安装依赖 ...
)
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败
    echo 国内网络可设置镜像后重试：
    echo   set HF_ENDPOINT=https://hf-mirror.com
    pause
    exit /b 1
)
echo.

REM 5) 检测 GPU，给出建议
echo [4/4] 检测 GPU ...
powershell -NoProfile -Command "$ErrorActionPreference='SilentlyContinue'; $g=Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }; if ($g) { Write-Host '   检测到 NVIDIA GPU：' $g.Name; Write-Host '   建议安装 CUDA 版 torch（RTX 30/40/50 系列必须 cu128+）：'; Write-Host '     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128' } else { Write-Host '   未检测到 NVIDIA GPU，使用 CPU 推理即可' }"
echo.

REM 6) 下一步提示
echo ============================================================
echo   ✅ 环境就绪（Python 3.13 + 依赖已装）
echo ============================================================
echo.
echo 接下来：
echo   1. 下载模型到项目 models/ 目录（首次约 92MB）：
echo        python scripts/download_models.py
echo.
echo   2. 启动 GUI：
echo        python main.py
echo.
echo   3. （可选）测试检索效果：
echo        python eval_retrieval.py
echo.
echo 若使用 PowerShell，每次启动需先激活 venv：
echo   .venv\Scripts\Activate.ps1
echo ============================================================
echo.
pause
