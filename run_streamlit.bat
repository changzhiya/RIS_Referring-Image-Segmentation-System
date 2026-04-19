@echo off
REM 在项目根目录启动 Streamlit。可先 set RIS_PYTHON=完整路径\python.exe
setlocal
cd /d "%~dp0"
if defined RIS_PYTHON (
  "%RIS_PYTHON%" -m streamlit run system\streamlit_app.py %*
) else (
  python -m streamlit run system\streamlit_app.py %*
)
endlocal
