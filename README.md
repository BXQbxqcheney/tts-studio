# TTS Studio · 本地中文声音克隆（开源脱敏版）

在本机运行中文声音克隆 TTS：上传 3–30 秒参考音频与逐字转写，即可用该音色生成中文语音；提供 WebUI 与 HTTP API。主引擎为 GPT-SoVITS v2ProPlus（底模 GPT + SoVITS），可选 Ming-3.0-TTS 零样本。

**作者**：白小谦
**反馈邮箱**：qianxlfk@163.com
**许可证**：AGPL-3.0-or-later（见 `LICENSE`）

> 本仓库不包含任何模型权重、用户录音、生成音频、训练日志或个人信息。模型需通过一键脚本从上游下载；音色数据由使用者自行提供并承担授权责任。

## 功能

- WebUI：上传参考音色、逐字转写、长文生成、段级进度、试听、下载、历史。
- HTTP API：短文本同步、10,000 字长任务、取消/恢复、直连 WAV、健康检查。
- 模型选择：`gptsovits`（长段）、`gptsovits_ft`（自备微调）、`ming`（需自行安装 Ming 运行时）。
- 资源模式：极限 / 高 / 较高 / 中 / 低 / 极低，按请求生效，无需重启。
- 并行与批处理：官方 batch 路径 + 可选双实例段级并行；质量闸门检测静音/截断。
- 安全：默认仅监听 127.0.0.1；非本机监听强制 `TTS_API_TOKEN`；Origin/Host 校验；上传与路径边界限制。

## 快速开始（Windows）

### 1. 安装服务环境

```powershell
setup_env.cmd
```

### 2. 一键下载模型（不含任何用户权重）

```powershell
download_models.cmd
```

- Ming 模型 + 官方代码：`models/Ming-3.0-tts-0.5B`、`third_party/Ming-3.0-tts-main`
- GPT-SoVITS v2ProPlus 整合包：`models/GPT-SoVITS/GPT-SoVITS-v2pro-20250604`（约 7GB）
- 可选 ASR 模型：`download_models.cmd --with-asr`

### 3. 准备 GPT-SoVITS 运行时

GPT-SoVITS 边车需要独立的 Python 运行时（建议 3.9，安装其 `requirements.txt`）。把整合包附带的 `runtime` 目录放到仓库根目录，或设置：

```powershell
set TTS_GPTSOVITS_RUNTIME=D:\path\to\python.exe
```

### 4. 启动

```powershell
start_tts.bat
```

浏览器打开 `http://127.0.0.1:8765`，上传自己的 3–30 秒参考音频与逐字转写。

## 数据与音色

- 不附带 `default` 或任何演示音色；首次使用请在「上传参考音色」中注册。
- 参考音频要求：同一人、安静环境、无 BGM/混响、逐字转写与音频完全一致；建议 5–10 秒。
- 只应克隆你拥有合法授权的声音；不要用于冒充、诈骗或侵犯他人权益。
- `data/` 与 `outputs/` 默认被 `.gitignore` 排除，不会进入仓库。

## 资源模式

| 模式 | 实例 | batch | 说明 |
|---|---|---|---|
| 极限 | 1 | 4 | 单实例 batch4，最快；显存约 7–8GB |
| 高 | 2 | 1 | 双实例段级并行；显存约 4.8GB |
| 较高 | 1 | 2 | 单实例 batch2；约 4.5GB |
| 中 | 1 | 1 | 约 2.4GB |
| 低 | 1 | 1 | 120 字安全块 |
| 极低 | 1 | 1 | 80 字安全块，最省峰值 |

说明：在 RTX 4060 Laptop 8GB 上，双实例 + batch≥2 会因显存争用反而变慢，因此“极限”采用单实例 batch4；“高”保留双实例并行。

## 性能（实测）

测试环境：RTX 4060 Laptop 8GB / i7-13650HX / 24GB RAM。

| 配置 | 文本 | 墙钟 | 音频 | RTF |
|---|---|---|---|---|
| 单实例 batch4（4k） | 4,000 字 / 4 段 | 213.2 秒 | 844.4 秒 | **0.249** |
| 单实例 batch4（离线 1k） | 1,000 字 | 26.6 秒 | 211.6 秒 | **0.126** |
| 双实例 batch1（4k） | 4,000 字 / 4 段 | 260.6 秒 | 836.0 秒 | **0.303** |
| 串行 160 字块 | 1,000 字 | 117 秒 | 205 秒 | 0.570 |

- 公开 GPT-SoVITS v2ProPlus 基准：RTX 4060Ti RTF 0.028、RTX 4090 RTF 0.014（官方 README）。
- **0.01 / 0.005 在 8GB RTX 4060 Laptop 上无法在不降质的前提下达到**；4090 官方也仅约 0.014。本仓库在当前硬件上以 batch4 尽量逼近硬件上限；如需 0.01 级，请在 4090/云端复测。
- 完整对比与消融见 [BENCHMARK.md](BENCHMARK.md)。

## HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 服务状态、显存、并行度、模型 |
| GET | `/api/engines` | 模型与资源模式 |
| GET/POST | `/api/voices` | 音色列表/上传 |
| POST | `/api/tts` | 同步生成（≤500 字，返回 JSON） |
| POST | `/api/tts/default` | 直连生成，返回 `audio/wav` |
| POST | `/api/tts/jobs` | 长任务（≤10,000 字） |
| GET | `/api/tts/jobs/{id}` | 长任务状态/段进度 |
| POST | `/api/tts/jobs/{id}/cancel` | 取消 |
| POST | `/api/tts/jobs/{id}/resume` | 恢复 |
| GET | `/api/history` | 最近生成 |

请求示例：

```json
{
  "text": "你好，这是本地声音克隆测试。",
  "voice_id": "你的音色ID",
  "model": "gptsovits",
  "mode": "extreme",
  "quality": "quality",
  "postprocess": "clean"
}
```

## 目录

```
app/                  FastAPI 服务、引擎、WebUI
scripts/              边车、数据准备、训练、模型下载
tests/                pytest
data/                 本地音色与参考段（git 忽略）
models/               模型（git 忽略，由 download_models.cmd 下载）
runtime/              GPT-SoVITS Python 运行时（git 忽略）
third_party/          Ming 官方代码（git 忽略）
download_models.cmd   一键下载模型
setup_env.cmd         安装服务环境
start_tts.bat         启动服务
BENCHMARK.md          对比与消融实验
LICENSE               AGPL-3.0-or-later
THIRD_PARTY_NOTICES.md
```

## 许可证与第三方

- 本项目代码：**AGPL-3.0-or-later**。修改/分发/通过网络提供服务时，必须按 AGPL 提供对应源码。
- GPT-SoVITS：MIT；Ming-3.0-TTS：MIT；faster-whisper、ModelScope 等见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 模型权重不随仓库分发，其许可与使用条件以上游为准。

## 作者

白小谦 · qianxlfk@163.com
