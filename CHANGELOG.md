# CHANGELOG

## 0.7.0 · 2026-11-01

- 开源脱敏版：移除模型权重、用户录音/转写/生成音频/训练日志与个人路径。
- 资源模式：极限/高/较高/中/低/极低，按请求生效。
- 官方 batch 推理路径：batch4 单实例 RTF 0.249（4k 服务实测）、离线 1k RTF 0.126。
- 双实例段级并行（高模式）：4k RTF 0.303，质量与单实例逐段一致。
- 长任务 10,000 字、段级进度、断点恢复、取消。
- 一键 `download_models.cmd`、`setup_env.cmd`、`start_tts.bat`。
- 所有代码以 AGPL-3.0-or-later 发布；作者：白小谦 <qianxlfk@163.com>。

## 0.6.0

- 模型可选：GPT-SoVITS 长段、自备微调 GPT、Ming 零样本。
- 外层 1000 字/段；边车 160/120 字安全块。
- 10k 字 10 段端到端。

## 0.5.0

- GPT-SoVITS v2ProPlus 边车、WebUI/API、长任务、音色注册。

## 0.1.0–0.4.0

- FastAPI 服务、WebUI、音色库、直连 WAV、长文本任务、并发与后处理。
