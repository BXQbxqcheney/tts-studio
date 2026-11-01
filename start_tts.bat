@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
set "TTS_ENGINE=gptsovits"
set "TTS_MODEL_DIR=%~dp0models\Ming-3.0-tts-0.5B"
set "TTS_CODE_DIR=%~dp0third_party\Ming-3.0-tts-main"
set "TTS_GPTSOVITS_PKG=%~dp0models\GPT-SoVITS\GPT-SoVITS-v2pro-20250604"
set "TTS_GPTSOVITS_RUNTIME=%~dp0runtime\python.exe"
set "TTS_GPTSOVITS_ASR_GATE=0"
set "TTS_RESOURCE_MODE=extreme"

if not exist ".venv\Scripts\python.exe" (
  echo 未找到 .venv，请先运行 setup_env.cmd。
  pause
  exit /b 1
)
if not exist "%TTS_GPTSOVITS_RUNTIME%" (
  echo 未找到 GPT-SoVITS 运行时：%TTS_GPTSOVITS_RUNTIME%
  echo 请把 GPT-SoVITS 的 runtime 目录放到仓库 runtime\，或设置 TTS_GPTSOVITS_RUNTIME。
  pause
  exit /b 1
)

powershell -NoProfile -Command "try { $r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8765/api/health' -TimeoutSec 2; if($r.StatusCode -eq 200){exit 0} } catch {}; exit 1"
if %errorlevel%==0 goto open

start "TTS Studio Server" /min ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8765
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8765/api/health' -TimeoutSec 2; if($r.StatusCode -eq 200){exit 0} } catch {}; Start-Sleep -Seconds 1 }; exit 1"
if %errorlevel% neq 0 (
  echo TTS Studio 启动失败，请查看 logs\server.log。
  pause
  exit /b 1
)

:open
start "" "http://127.0.0.1:8765"
endlocal
