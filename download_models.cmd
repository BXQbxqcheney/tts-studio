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
  echo [1/2] 创建下载环境 .venv ...
  python -m venv .venv
)
echo [2/2] 安装下载依赖并拉取模型 ...
".venv\Scripts\python.exe" -m pip install -U pip
".venv\Scripts\python.exe" -m pip install modelscope py7zr requests
".venv\Scripts\python.exe" "scripts\download_models_public.py" --dest "%~dp0models" %*
if errorlevel 1 (
  echo.
  echo 下载未完全成功，请查看上方错误或 models\MODELS_READY.json。
) else (
  echo.
  echo 模型下载完成：%~dp0models
)
pause
endlocal
