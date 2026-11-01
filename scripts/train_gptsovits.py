"""GPT-SoVITS v2ProPlus 最高质量训练流水线：数据准备 → SoVITS → GPT。

使用 Temp runtime（解决 F 盘 DLL 写入限制）；数据/权重仍在 F 盘。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import yaml

PKG = Path(r"models/GPT-SoVITS/GPT-SoVITS-v2pro-20250604")
RUNTIME = Path(r"runtime/python.exe")
EXP = PKG / "logs" / "myvoice"
LIST = Path(r".\data\gpt_sovits\myvoice.list")
WAV_DIR = Path(r".\data\processed\myvoice")
LOG = Path(r".\logs\gpt_sovits_train.log")
STATUS = Path(r"models\GPT-SoVITS\training_status.json")
DONE = Path(r"models\GPT-SoVITS\TRAINING_DONE.json")

BERT = PKG / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
HUBERT = PKG / "GPT_SoVITS/pretrained_models/chinese-hubert-base"
SV = PKG / "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
S2G = PKG / "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
S2D = PKG / "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth"
S1 = PKG / "GPT_SoVITS/pretrained_models/s1v3.ckpt"
S2_CONFIG = PKG / "GPT_SoVITS/configs/s2v2ProPlus.json"
S1_CONFIG = PKG / "GPT_SoVITS/configs/s1longer-v2.yaml"
SOVITS_OUT = PKG / "SoVITS_weights_v2ProPlus"
GPT_OUT = PKG / "GPT_weights_v2ProPlus"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(stage: str, status: str, **extra) -> None:
    data = {
        "stage": stage,
        "status": status,
        "updated_at": now(),
        "exp_dir": str(EXP),
        "log": str(LOG),
    }
    data.update(extra)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_stage(name: str, command: list[str], cwd: Path, env: dict[str, str], log) -> None:
    write_status(name, "running")
    log.write(f"\n\n===== {name} {now()} =====\n$ {' '.join(command)}\n")
    log.flush()
    print(f"[{now()}] {name} start", flush=True)
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log.flush()
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed rc={result.returncode}")
    print(f"[{now()}] {name} done", flush=True)
    write_status(name, "done")


def latest_file(directory: Path, patterns: list[str], started: float) -> Path | None:
    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    files = [p for p in files if p.is_file() and p.stat().st_mtime >= started - 5]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().timestamp()
    try:
        for path in (RUNTIME, LIST, WAV_DIR, BERT, HUBERT, SV, S2G, S2D, S1, S2_CONFIG, S1_CONFIG):
            if not path.exists():
                raise RuntimeError(f"missing required path: {path}")
        EXP.mkdir(parents=True, exist_ok=True)
        (EXP / "logs_s2_v2ProPlus").mkdir(parents=True, exist_ok=True)
        (EXP / "logs_s1_v2ProPlus").mkdir(parents=True, exist_ok=True)
        SOVITS_OUT.mkdir(parents=True, exist_ok=True)
        GPT_OUT.mkdir(parents=True, exist_ok=True)
        base_env = os_environ()
        base_env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "CUDA_VISIBLE_DEVICES": "0",
                "_CUDA_VISIBLE_DEVICES": "0",
                "is_half": "True",
                "version": "v2Pro",
            }
        )
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n\n########## GPT-SoVITS training {now()} ##########\n")
            prepare_env = dict(base_env)
            prepare_env.update(
                {
                    "inp_text": str(LIST),
                    "inp_wav_dir": str(WAV_DIR),
                    "exp_name": "myvoice",
                    "i_part": "0",
                    "all_parts": "1",
                    "opt_dir": str(EXP),
                }
            )
            prepare_cwd = PKG

            text_env = dict(prepare_env)
            text_env["bert_pretrained_dir"] = str(BERT)
            run_stage("1a_text", [str(RUNTIME), "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], prepare_cwd, text_env, log)

            hubert_env = dict(prepare_env)
            hubert_env.update({"cnhubert_base_dir": str(HUBERT), "sv_path": str(SV)})
            run_stage("1b_hubert", [str(RUNTIME), "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], prepare_cwd, hubert_env, log)
            run_stage("1b_sv", [str(RUNTIME), "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"], prepare_cwd, hubert_env, log)

            sem_env = dict(prepare_env)
            sem_env.update({"pretrained_s2G": str(S2G), "s2config_path": str(S2_CONFIG)})
            run_stage("1c_semantic", [str(RUNTIME), "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], prepare_cwd, sem_env, log)

        part_text = EXP / "2-name2text-0.txt"
        final_text = EXP / "2-name2text.txt"
        if not part_text.is_file():
            raise RuntimeError(f"missing {part_text}")
        shutil.copyfile(part_text, final_text)
        part_sem = EXP / "6-name2semantic-0.tsv"
        if not part_sem.is_file():
            raise RuntimeError(f"missing {part_sem}")
        sem_lines = [line for line in part_sem.read_text(encoding="utf-8").splitlines() if line.strip()]
        (EXP / "6-name2semantic.tsv").write_text(
            "item_name\tsemantic_audio\n" + "\n".join(sem_lines) + "\n",
            encoding="utf-8",
        )
        counts = {
            "name2text": len(final_text.read_text(encoding="utf-8").splitlines()),
            "semantic": len(sem_lines),
            "cnhubert": len(list((EXP / "4-cnhubert").glob("*.pt"))),
            "wav32k": len(list((EXP / "5-wav32k").glob("*.wav"))),
            "sv_cn": len(list((EXP / "7-sv_cn").glob("*.pt"))),
        }
        if min(counts.values()) < 40:
            raise RuntimeError(f"dataset prep incomplete: {counts}")
        write_status("1_prepare", "done", counts=counts)

        with LOG.open("a", encoding="utf-8") as log:
            s2_data = json.loads(S2_CONFIG.read_text(encoding="utf-8"))
            s2_data["train"].update(
                {
                    "batch_size": 4,
                    "epochs": 20,
                    "text_low_lr_rate": 0.4,
                    "pretrained_s2G": str(S2G),
                    "pretrained_s2D": str(S2D),
                    "if_save_latest": True,
                    "if_save_every_weights": True,
                    "save_every_epoch": 1,
                    "gpu_numbers": "0",
                    "grad_ckpt": True,
                    "lora_rank": 0,
                    "fp16_run": True,
                }
            )
            s2_data["model"]["version"] = "v2ProPlus"
            s2_data["data"]["exp_dir"] = str(EXP)
            s2_data["s2_ckpt_dir"] = str(EXP)
            s2_data["save_weight_dir"] = str(SOVITS_OUT)
            s2_data["name"] = "myvoice"
            s2_data["version"] = "v2ProPlus"
            s2_config_path = EXP / "s2_config.json"
            s2_config_path.write_text(json.dumps(s2_data, ensure_ascii=False, indent=2), encoding="utf-8")
            s2_started = datetime.now().timestamp()
            run_stage(
                "2_sovits_train",
                [str(RUNTIME), "-s", "GPT_SoVITS/s2_train.py", "--config", str(s2_config_path)],
                PKG,
                base_env,
                log,
            )

        with LOG.open("a", encoding="utf-8") as log:
            s1_data = yaml.safe_load(S1_CONFIG.read_text(encoding="utf-8"))
            s1_data["train"].update(
                {
                    "batch_size": 4,
                    "epochs": 20,
                    "precision": "16-mixed",
                    "save_every_n_epoch": 1,
                    "if_save_every_weights": True,
                    "if_save_latest": True,
                    "if_dpo": False,
                    "half_weights_save_dir": str(GPT_OUT),
                    "exp_name": "myvoice",
                }
            )
            s1_data["pretrained_s1"] = str(S1)
            s1_data["train_semantic_path"] = str(EXP / "6-name2semantic.tsv")
            s1_data["train_phoneme_path"] = str(EXP / "2-name2text.txt")
            s1_data["output_dir"] = str(EXP / "logs_s1_v2ProPlus")
            s1_config_path = EXP / "s1_config.yaml"
            s1_config_path.write_text(yaml.safe_dump(s1_data, allow_unicode=True), encoding="utf-8")
            s1_started = datetime.now().timestamp()
            run_stage(
                "3_gpt_train",
                [str(RUNTIME), "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(s1_config_path)],
                PKG,
                base_env,
                log,
            )

        sovits_weight = latest_file(SOVITS_OUT, ["*.pth"], s2_started)
        gpt_weight = latest_file(GPT_OUT, ["*.ckpt", "*.pth"], s1_started)
        if sovits_weight is None or gpt_weight is None:
            raise RuntimeError(f"trained weights not found: sovits={sovits_weight} gpt={gpt_weight}")
        DONE.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "exp_dir": str(EXP),
                    "sovits_weight": str(sovits_weight),
                    "gpt_weight": str(gpt_weight),
                    "counts": counts,
                    "s2_config": str(EXP / "s2_config.json"),
                    "s1_config": str(EXP / "s1_config.yaml"),
                    "finished_at": now(),
                    "log": str(LOG),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_status("done", "ok", sovits_weight=str(sovits_weight), gpt_weight=str(gpt_weight))
        print("TRAINING OK", sovits_weight, gpt_weight, flush=True)
        return 0
    except Exception as exc:
        DONE.write_text(
            json.dumps(
                {
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "finished_at": now(),
                    "log": str(LOG),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_status("error", "error", error=repr(exc))
        traceback.print_exc()
        return 1


def os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
