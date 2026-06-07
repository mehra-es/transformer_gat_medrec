"""Background pipeline runner for UI (mirrors run.sh steps)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]


class StepStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepState:
    id: str
    title: str
    description: str
    command: str
    status: StepStatus = StepStatus.IDLE
    last_message: str = ""
    last_run_at: Optional[float] = None
    duration_sec: Optional[float] = None


class PipelineRunner:
    """Runs setup / train / eval / test / explain subprocesses with logging."""

    STEPS_META = [
        ("setup", "Environment setup", "Create venv (if needed) and install requirements.txt"),
        ("train", "Train model", "Fit Transformer-GAT on synthetic EHR data"),
        ("eval", "Evaluate", "Report test-set Jaccard, F1, PRAUC, DDI rate"),
        ("test", "Unit tests", "Run pytest on model, loss, and metrics"),
        ("explain", "SHAP explain", "DeepSHAP attributions for a demo patient"),
        ("explore", "Explore results", "Use dashboard views (patient, fusion, metrics)"),
    ]

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT
        self.log_dir = self.root / "logs" / "pipeline"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = "config.yaml"
        self.checkpoint = self.root / "checkpoints" / "best.pt"
        self.use_synthetic = "true"
        self._lock = threading.Lock()
        self._job_thread: Optional[threading.Thread] = None
        self._current_job: Optional[str] = None
        self._stop_requested = False
        self._on_complete: Optional[Callable[[], None]] = None
        self.steps: Dict[str, StepState] = {
            sid: StepState(
                id=sid,
                title=title,
                description=desc,
                command=self._command_preview(sid),
            )
            for sid, title, desc in self.STEPS_META
        }
        self.steps["explore"].status = StepStatus.SUCCESS
        self.steps["explore"].last_message = "You are using the dashboard."

    def set_reload_callback(self, cb: Callable[[], None]) -> None:
        self._on_complete = cb

    def _python(self) -> Path:
        venv_py = self.root / ".venv" / "bin" / "python"
        if venv_py.exists():
            return venv_py
        return Path(sys.executable)

    def _pip(self) -> List[str]:
        return [str(self._python()), "-m", "pip"]

    def _command_preview(self, step_id: str) -> str:
        py = ".venv/bin/python" if (self.root / ".venv" / "bin" / "python").exists() else "python3"
        cmds = {
            "setup": f"{py} -m pip install -r requirements.txt",
            "train": f"{py} src/main.py --config config.yaml --mode train --use_synthetic true",
            "eval": f"{py} src/evaluate.py --checkpoint checkpoints/best.pt --use_synthetic true",
            "test": f"{py} -m pytest tests/ -v",
            "explain": f"{py} src/explain.py --checkpoint checkpoints/best.pt --use_synthetic true --patient_idx 0",
            "explore": "Open sidebar views: Patient → Recommendations → Fusion → Metrics",
        }
        return cmds.get(step_id, "")

    def _build_cmd(self, step_id: str, patient_idx: int = 0) -> List[str]:
        py = str(self._python())
        if step_id == "setup":
            return self._pip() + ["install", "-q", "-r", str(self.root / "requirements.txt")]
        if step_id == "train":
            return [
                py,
                str(self.root / "src" / "main.py"),
                "--config",
                self.config,
                "--mode",
                "train",
                "--use_synthetic",
                self.use_synthetic,
            ]
        if step_id == "eval":
            return [
                py,
                str(self.root / "src" / "evaluate.py"),
                "--checkpoint",
                "checkpoints/best.pt",
                "--use_synthetic",
                self.use_synthetic,
            ]
        if step_id == "test":
            return [py, "-m", "pytest", "tests/", "-v", "--tb=short"]
        if step_id == "explain":
            return [
                py,
                str(self.root / "src" / "explain.py"),
                "--checkpoint",
                "checkpoints/best.pt",
                "--use_synthetic",
                self.use_synthetic,
                "--patient_idx",
                str(patient_idx),
            ]
        raise ValueError(f"Unknown step: {step_id}")

    def check_setup(self) -> Dict[str, Any]:
        venv_ok = (self.root / ".venv" / "bin" / "python").exists()
        req_ok = (self.root / "requirements.txt").exists()
        torch_ok = False
        if venv_ok:
            try:
                subprocess.run(
                    [str(self._python()), "-c", "import torch; import torch_geometric"],
                    cwd=self.root,
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
                torch_ok = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        return {
            "venv_exists": venv_ok,
            "requirements_file": req_ok,
            "dependencies_ok": torch_ok,
            "ready": venv_ok and torch_ok,
        }

    def is_busy(self) -> bool:
        return self._job_thread is not None and self._job_thread.is_alive()

    def _run_subprocess(self, step_id: str, patient_idx: int = 0) -> bool:
        step = self.steps[step_id]
        log_path = self.log_dir / f"{step_id}.log"
        cmd = self._build_cmd(step_id, patient_idx)
        step.status = StepStatus.RUNNING
        step.last_message = "Running…"
        step.command = " ".join(cmd)
        t0 = time.time()

        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(cmd)}\n\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if self._stop_requested:
                    proc.terminate()
                    break
                logf.write(line)
                logf.flush()
            rc = proc.wait()

        step.duration_sec = round(time.time() - t0, 2)
        step.last_run_at = time.time()
        if self._stop_requested:
            step.status = StepStatus.FAILED
            step.last_message = "Cancelled"
            return False
        if rc == 0:
            step.status = StepStatus.SUCCESS
            step.last_message = f"Completed in {step.duration_sec}s"
            return True
        step.status = StepStatus.FAILED
        step.last_message = f"Exit code {rc} — see log"
        return False

    def run_step(
        self,
        step_id: str,
        *,
        skip_train_if_checkpoint: bool = False,
        patient_idx: int = 0,
    ) -> bool:
        if step_id == "explore":
            return True
        if step_id not in self.steps:
            raise ValueError(step_id)

        if step_id == "train" and skip_train_if_checkpoint and self.checkpoint.exists():
            s = self.steps["train"]
            s.status = StepStatus.SKIPPED
            s.last_message = "Checkpoint already exists (skip enabled)"
            s.last_run_at = time.time()
            return True

        if step_id == "eval" and not self.checkpoint.exists():
            self.steps["eval"].status = StepStatus.FAILED
            self.steps["eval"].last_message = "Train first — no checkpoint"
            return False

        ok = self._run_subprocess(step_id, patient_idx)
        if ok and step_id in ("train", "eval") and self._on_complete:
            self._on_complete()
        return ok

    def _job_run_all(self, skip_train: bool, patient_idx: int) -> None:
        order = ["setup", "train", "eval", "test", "explain"]
        self._current_job = "all"
        for sid in order:
            if self._stop_requested:
                break
            if sid == "train":
                self.run_step(sid, skip_train_if_checkpoint=skip_train, patient_idx=patient_idx)
            else:
                self.run_step(sid, patient_idx=patient_idx)
        self._current_job = None

    def start_step(
        self,
        step_id: str,
        *,
        skip_train_if_checkpoint: bool = False,
        patient_idx: int = 0,
    ) -> Dict[str, Any]:
        if self.is_busy():
            return {"ok": False, "error": "Another job is already running"}
        if step_id == "all":
            self._stop_requested = False
            self._job_thread = threading.Thread(
                target=self._job_run_all,
                kwargs={"skip_train": skip_train_if_checkpoint, "patient_idx": patient_idx},
                daemon=True,
            )
            self._job_thread.start()
            return {"ok": True, "message": "Started full pipeline"}

        self._stop_requested = False

        def _target() -> None:
            self._current_job = step_id
            self.run_step(
                step_id,
                skip_train_if_checkpoint=skip_train_if_checkpoint,
                patient_idx=patient_idx,
            )
            self._current_job = None

        self._job_thread = threading.Thread(target=_target, daemon=True)
        self._job_thread.start()
        return {"ok": True, "message": f"Started {step_id}"}

    def cancel(self) -> Dict[str, Any]:
        self._stop_requested = True
        return {"ok": True, "message": "Cancel requested"}

    def get_logs(self, step_id: str, tail: int = 200) -> str:
        path = self.log_dir / f"{step_id}.log"
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    def to_dict(self) -> Dict[str, Any]:
        setup = self.check_setup()
        return {
            "busy": self.is_busy(),
            "current_job": self._current_job,
            "checkpoint_exists": self.checkpoint.exists(),
            "setup": setup,
            "steps": [
                {
                    "id": s.id,
                    "title": s.title,
                    "description": s.description,
                    "command": s.command,
                    "status": s.status.value,
                    "last_message": s.last_message,
                    "last_run_at": s.last_run_at,
                    "duration_sec": s.duration_sec,
                }
                for s in self.steps.values()
            ],
            "step_order": ["setup", "train", "eval", "test", "explain", "explore"],
        }


_runner: Optional[PipelineRunner] = None


def get_runner() -> PipelineRunner:
    global _runner
    if _runner is None:
        _runner = PipelineRunner()
    return _runner
