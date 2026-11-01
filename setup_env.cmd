@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] 未找到 Python 3.11+，请先安装 Python 并加入 PATH。
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo 创建服务环境 .venv ...
  python -m venv .venv
)
echo 安装服务依赖 ...
".venv\Scripts\python.exe" -m pip install -U pip
".venv\Scripts\python.exe" -m pip install -e .
".venv\Scripts\python.exe" -m pip install pytest httpx
echo.
echo 服务环境完成：.venv
echo 下一步：
echo   1) 运行 download_models.cmd 下载模型
echo   2) 准备 GPT-SoVITS 的 Python 运行时并设置 TTS_GPTSOVITS_RUNTIME
echo   3) 运行 start_tts.bat
pause
endlocal
