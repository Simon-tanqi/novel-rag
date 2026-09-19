# ragas_env_setup.ps1 — 创建 RAGAS 独立评测环境（不污染主 .venv）
#
# 用法（在项目根目录执行）：
#     powershell -ExecutionPolicy Bypass -File scripts/ragas_env_setup.ps1
#     powershell -ExecutionPolicy Bypass -File scripts/ragas_env_setup.ps1 -Mirror https://pypi.tuna.tsinghua.edu.cn/simple
#
# 完成后：
#     .\.ragas_venv\Scripts\python.exe scripts/ragas_eval.py dry-run --dataset ragas_out/zhetian_natural_dataset.json
#     .\.ragas_venv\Scripts\python.exe scripts/ragas_eval.py score --dataset ragas_out/zhetian_natural_dataset.json

param(
    [string]$VenvDir = ".ragas_venv",
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cpu",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== 1/4 创建独立虚拟环境 $VenvDir ==" -ForegroundColor Cyan
if (Test-Path $VenvDir) {
    Write-Host "已存在，跳过创建：$VenvDir" -ForegroundColor Yellow
} else {
    python -m venv $VenvDir
}

$Py = Join-Path $Root "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "未找到解释器: $Py" }

Write-Host "== 2/4 升级 pip ==" -ForegroundColor Cyan
& $Py -m pip install --upgrade pip -i $Mirror

if (-not $SkipTorch) {
    Write-Host "== 3/4 安装 CPU 版 torch（RAGAS 的 sentence-transformers 依赖）==" -ForegroundColor Cyan
    & $Py -m pip install torch --index-url $TorchIndex
}

Write-Host "== 4/4 安装 ragas 与兼容 langchain 版本簇 ==" -ForegroundColor Cyan
& $Py -m pip install -r (Join-Path $Root "requirements-extras-ragas.txt") -i $Mirror

Write-Host "== 自检 ==" -ForegroundColor Cyan
& $Py -c "import ragas, langchain, langchain_community, sentence_transformers; print('ragas', ragas.__version__); print('langchain', langchain.__version__); print('OK')"
Write-Host "完成。评测命令见 README 的「RAGAS 系统评测」章节。" -ForegroundColor Green
