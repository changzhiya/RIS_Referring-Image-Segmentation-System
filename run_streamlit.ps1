# 在项目根目录启动 Streamlit 交互推理界面。
# 用法（PowerShell）:
#   .\run_streamlit.ps1
#   .\run_streamlit.ps1 -- --server.port 8502
# 可选环境变量:
#   $env:RIS_PYTHON = "D:\miniconda3\envs\ris_env\python.exe"   # 指定解释器；未设置则用 PATH 中的 python
#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$py = if ($env:RIS_PYTHON) { $env:RIS_PYTHON } else { "python" }
$app = Join-Path $ProjectRoot "system\streamlit_app.py"
& $py -m streamlit run $app @args
