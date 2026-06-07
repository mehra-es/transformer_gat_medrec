"""FastAPI server for the MedRec dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.inference import InferenceService
from src.ui.pipeline import get_runner

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Transformer-GAT MedRec",
    description="Clinical decision support dashboard (not autonomous prescribing).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_service: InferenceService | None = None


def reload_inference_service() -> None:
    """Reload model after training completes."""
    global _service
    _service = None
    get_service()


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


@app.on_event("startup")
def startup() -> None:
    runner = get_runner()
    runner.set_reload_callback(reload_inference_service)
    try:
        get_service()
    except Exception:
        pass


class PipelineRunBody(BaseModel):
    skip_train_if_checkpoint: bool = False
    patient_idx: int = 0


@app.get("/api/pipeline")
def api_pipeline() -> dict:
    return get_runner().to_dict()


@app.get("/api/pipeline/logs/{step_id}")
def api_pipeline_logs(step_id: str, tail: int = Query(300, ge=10, le=2000)) -> dict:
    if step_id not in get_runner().steps and step_id != "all":
        raise HTTPException(404, "Unknown step")
    log_step = "train" if step_id == "all" else step_id
    return {"step_id": step_id, "log": get_runner().get_logs(log_step, tail=tail)}


@app.post("/api/pipeline/run/{step_id}")
def api_pipeline_run(step_id: str, body: PipelineRunBody | None = None) -> dict:
    body = body or PipelineRunBody()
    runner = get_runner()
    if step_id not in (*runner.steps.keys(), "all"):
        raise HTTPException(404, f"Unknown step: {step_id}")
    result = runner.start_step(
        step_id,
        skip_train_if_checkpoint=body.skip_train_if_checkpoint,
        patient_idx=body.patient_idx,
    )
    if not result.get("ok"):
        raise HTTPException(409, result.get("error", "Cannot start job"))
    return result


@app.post("/api/pipeline/cancel")
def api_pipeline_cancel() -> dict:
    return get_runner().cancel()


@app.post("/api/pipeline/reload-model")
def api_reload_model() -> dict:
    reload_inference_service()
    return get_service().status()


@app.get("/api/status")
def api_status() -> dict:
    return get_service().status()


@app.get("/api/patients")
def api_patients() -> dict:
    svc = get_service()
    patients = [
        {"index": i, "patient_id": svc.dataset[i]["patient_id"]}
        for i in range(svc.num_patients)
    ]
    return {"patients": patients}


@app.get("/api/patient/{idx}")
def api_patient(idx: int) -> dict:
    svc = get_service()
    if idx < 0 or idx >= svc.num_patients:
        raise HTTPException(404, "Patient not found")
    return svc.patient_summary(idx)


@app.get("/api/predict/{idx}")
def api_predict(idx: int, top_k: int = 15) -> dict:
    svc = get_service()
    if idx < 0 or idx >= svc.num_patients:
        raise HTTPException(404, "Patient not found")
    return svc.predict(idx, top_k=top_k)


@app.get("/api/metrics")
def api_metrics() -> dict:
    svc = get_service()
    if not svc.model_loaded:
        return {"error": "No checkpoint loaded", "metrics": None}
    return {"metrics": svc.evaluate_test_set()}


@app.get("/api/explain/{idx}")
def api_explain(idx: int) -> dict:
    svc = get_service()
    if idx < 0 or idx >= svc.num_patients:
        raise HTTPException(404, "Patient not found")
    result = svc.explain(idx)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "ui.server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()
