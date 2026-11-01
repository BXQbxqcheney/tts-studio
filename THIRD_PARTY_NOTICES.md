# 第三方组件与许可

本项目自身以 AGPL-3.0-or-later 发布。以下上游组件/模型不随仓库分发，使用者需自行下载并遵守其许可：

| 组件 | 用途 | 许可 | 上游 |
|---|---|---|---|
| GPT-SoVITS | 主 TTS 引擎/训练/推理 | MIT | https://github.com/RVC-Boss/GPT-SoVITS |
| Ming-3.0-TTS | 零样本 TTS 可选引擎 | MIT | https://github.com/inclusionAI/Ming-3.0-tts |
| faster-whisper | 可选 ASR 内容校验 | MIT | https://github.com/SYSTRAN/faster-whisper |
| ModelScope | 模型下载/托管 | Apache-2.0（工具） | https://www.modelscope.cn |
| PyTorch / torchaudio | 推理运行时 | BSD-3-Clause | https://pytorch.org |
| FastAPI / Uvicorn / Pydantic | HTTP 服务 | MIT / BSD / MIT | 各自项目 |
| ffmpeg | 音频后处理 | LGPL/GPL（视构建） | https://ffmpeg.org |

模型权重（Ming、GPT-SoVITS、ASR）的许可与使用范围以上游模型卡为准；本项目不授予任何模型权重许可。

声音克隆仅可用于你拥有合法授权的声音。禁止用于冒充他人、诈骗、骚扰、伪造证据或其他违法用途。
