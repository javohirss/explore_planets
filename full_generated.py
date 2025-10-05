# FastAPI backend for Novice mode (single-file demo)
# -------------------------------------------------
# Implements the routes described earlier:
# - Global: /health, /novice/meta, /about
# - Glossary: /glossary, /glossary/{slug}
# - Learn: /novice/learn/chapters, /novice/learn/chapters/{slug}, POST /novice/learn/chapters/{slug}/quiz
# - Explore: POST /novice/explore/simulate
# - Predict: POST /novice/predict (JSON or multipart CSV), GET /novice/predict/{id},
#            GET /novice/predict/{id}/download?format=png|pdf
#
# Notes:
# - This is a self-contained demo: uses local JSON-like dicts for content storage
#   and a simple file-based store for prediction results under ./.store/results.
# - Replace stubs with real content/DB in production.

from __future__ import annotations

import io
import json
import math
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Optional pandas import for CSV; fallback to csv module if missing
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore
import csv

# Matplotlib for PNG/PDF export (Agg backend)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# App bootstrap & config
# -----------------------------

app = FastAPI(title="Exoplanet Novice API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORE_DIR = Path(".store/results")
STORE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Static demo content (stubs)
# -----------------------------

ABOUT = {
    "title": "About the Project",
    "body": "This demo explains how transit detection works and offers a simple predictor.",
    "credits": ["Kepler", "K2", "TESS"],
    "ethics": "Educational use only. Do not use for real mission operations.",
}

GLOSSARY = {
    "transit": {
        "term": "Transit",
        "definition": "A dip in a star's brightness when a planet passes in front of it.",
        "mini_diagram": "/assets/diagrams/transit.svg",
    },
    "light-curve": {
        "term": "Light Curve",
        "definition": "Brightness vs. time measurement for a star.",
        "mini_diagram": "/assets/diagrams/lightcurve.svg",
    },
}

LEARN_CHAPTERS = [
    {
        "slug": "what-is-exoplanet",
        "title": "What is an exoplanet?",
        "summary": "Planets orbiting other stars.",
        "est_read_min": 1,
        "has_quiz": True,
        "content": """
        <h2>Exoplanets</h2>
        <p>Planets outside our Solar System. Thousands have been discovered.</p>
        """,
        "quiz": {
            "questions": [
                {
                    "id": "q1",
                    "text": "An exoplanet orbits...",
                    "options": ["its host star", "the Earth", "the Moon"],
                    "answer": 0,
                    "explain": "By definition, an exoplanet orbits a star other than the Sun.",
                }
            ]
        },
    },
    {
        "slug": "how-transit-works",
        "title": "How transit detection works",
        "summary": "Periodic brightness dips indicate a planet.",
        "est_read_min": 2,
        "has_quiz": False,
        "content": """
        <h2>Transit Method</h2>
        <p>We look for periodic dips in a light curve when a planet transits.</p>
        """,
    },
]

NOVICE_META = {
    "disclaimer": "Educational use only. Results are illustrative.",
    "version": "2025-10-02",
    "links": {
        "sample_csv": "/samples/lightcurve.csv",
    },
}

# -----------------------------
# Schemas
# -----------------------------

class Message(BaseModel):
    message: str


# Explore
class SimulateRequest(BaseModel):
    depth: float = Field(..., ge=0.0, le=0.1, description="Transit depth (fraction), e.g., 0.01 = 1%")
    duration: float = Field(..., gt=0.0, description="Transit duration in hours")
    period: float = Field(..., gt=0.0, description="Orbital period in days")
    noise: float = Field(0.0, ge=0.0, description="Gaussian noise std dev")
    length: float = Field(30.0, gt=0.0, description="Total simulated time in days")
    cadence: float = Field(0.02, gt=0.0, description="Sampling cadence in days (e.g., 0.02 ~ 28.8 min)")
    seed: Optional[int] = None


class SimulateResponse(BaseModel):
    time: list[float]
    flux: list[float]
    expected: list[float]
    meta: dict[str, Any]


# Predict
class Series(BaseModel):
    t: list[float]
    y: list[float]


class PredictJSONRequest(BaseModel):
    series: list[Series]
    metadata: Optional[dict[str, Any]] = None


class PredictResponse(BaseModel):
    result_id: str
    probability: float
    label: Literal["Low", "Med", "High"]
    uncertainty_note: str
    preview: dict[str, Any]
    features: dict[str, float]


# -----------------------------
# Utilities: storage for results
# -----------------------------

def _result_path(rid: str) -> Path:
    return STORE_DIR / f"{rid}.json"


def save_result(payload: dict[str, Any]) -> str:
    rid = uuid.uuid7().hex  # type: ignore[attr-defined]
    path = _result_path(rid)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"id": rid, **payload}, f)
    return rid


def load_result(rid: str) -> dict[str, Any]:
    path = _result_path(rid)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Services: simulator & predictor
# -----------------------------

def simulate_light_curve(req: SimulateRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if req.seed is not None:
        np.random.seed(req.seed)

    # Time grid
    t = np.arange(0.0, req.length, req.cadence, dtype=float)
    n = t.size

    # Expected model (box-shaped transit repeated with period)
    flux_expected = np.ones_like(t)
    # Convert duration hours -> fraction of day
    dur_days = req.duration / 24.0

    # Place a transit centered at phase = 0 for each period multiple
    phase = np.mod(t, req.period)
    in_transit = (phase < dur_days / 2.0) | (phase > req.period - dur_days / 2.0)
    flux_expected[in_transit] -= req.depth

    # Add noise
    noise = np.random.normal(0.0, req.noise, size=n)
    flux_observed = flux_expected + noise

    # Meta
    sigma = float(np.median(np.abs(flux_observed - np.median(flux_observed))) * 1.4826)
    snr = float((req.depth / (sigma + 1e-9)) * math.sqrt(max(1, int(dur_days / req.cadence))))
    meta = {"snr": snr, "num_points": int(n)}

    return t, flux_observed, flux_expected, meta


def downsample(time: np.ndarray, flux: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    n = time.size
    if n <= max_points:
        return time, flux
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return time[idx], flux[idx]


def score_series(time: np.ndarray, flux: np.ndarray) -> dict[str, Any]:
    # Basic validation
    if time.size < 200:
        raise HTTPException(status_code=422, detail="Need at least 200 points")

    # Sort by time
    order = np.argsort(time)
    t = time[order]
    y = flux[order]

    # Normalize around median ~ 1.0
    y = y / np.median(y)

    # Estimate noise via MAD
    sigma = float(np.median(np.abs(y - np.median(y))) * 1.4826)

    # Moving median to find dips
    win = max(5, int(0.01 * y.size))  # ~1% of series length
    if win % 2 == 0:
        win += 1

    # Simple rolling median via convolution approximation
    kernel = np.ones(win) / win
    y_smooth = np.convolve(y, kernel, mode="same")

    # Candidate dip
    dip_idx = int(np.argmin(y_smooth))
    baseline = 1.0
    Ahat = float(max(0.0, baseline - y_smooth[dip_idx]))  # estimated depth

    # Estimate duration (contiguous region below (1 - Ahat/2))
    thresh = baseline - Ahat / 2.0
    left = dip_idx
    while left > 0 and y_smooth[left] < thresh:
        left -= 1
    right = dip_idx
    while right < y_smooth.size - 1 and y_smooth[right] < thresh:
        right += 1
    Dhat = float(max(1, right - left))  # in points

    # Signal score
    S = float((Ahat / (sigma + 1e-9)) * math.sqrt(Dhat))

    # Logistic mapping to probability
    a, b = 0.9, -1.5
    p = 1.0 / (1.0 + math.exp(-(a * S + b)))

    # Label
    label = "Low"
    if p >= 0.66:
        label = "High"
    elif p >= 0.33:
        label = "Med"

    note = "Low noise" if sigma < 0.002 else ("Moderate noise" if sigma < 0.01 else "High noise")

    return {
        "probability": p,
        "label": label,
        "features": {"Ahat": Ahat, "Dhat": Dhat, "sigma": sigma, "S": S},
        "uncertainty_note": f"{note}; score driven by depth {Ahat:.4f} and duration ~{Dhat:.0f} points.",
        "preview": None,  # will be filled by caller
    }


# -----------------------------
# Routers / Endpoints
# -----------------------------

@app.get("/health", response_model=Message)
def health() -> Message:
    return Message(message="ok")


@app.get("/novice/meta")
def novice_meta() -> dict[str, Any]:
    return NOVICE_META


@app.get("/about")
def about() -> dict[str, Any]:
    return ABOUT


@app.get("/glossary")
def glossary_list() -> list[dict[str, Any]]:
    return [GLOSSARY[k] | {"slug": k} for k in sorted(GLOSSARY.keys())]


@app.get("/glossary/{slug}")
def glossary_item(slug: str) -> dict[str, Any]:
    item = GLOSSARY.get(slug)
    if not item:
        raise HTTPException(status_code=404, detail="Term not found")
    return item | {"slug": slug}


@app.get("/novice/learn/chapters")
def learn_chapters() -> list[dict[str, Any]]:
    # Return the brief listing
    return [
        {
            "slug": ch["slug"],
            "title": ch["title"],
            "summary": ch["summary"],
            "est_read_min": ch.get("est_read_min", 1),
            "has_quiz": ch.get("has_quiz", False),
        }
        for ch in LEARN_CHAPTERS
    ]


@app.get("/novice/learn/chapters/{slug}")
def learn_chapter(slug: str) -> dict[str, Any]:
    for ch in LEARN_CHAPTERS:
        if ch["slug"] == slug:
            body = {
                k: ch[k]
                for k in ["slug", "title", "summary", "est_read_min", "has_quiz", "content"]
                if k in ch
            }
            if ch.get("has_quiz") and ch.get("quiz"):
                # Do not expose correct answers directly; only questions
                qs = ch["quiz"]["questions"]
                body["quiz"] = {
                    "questions": [
                        {"id": q["id"], "text": q["text"], "options": q["options"]}
                        for q in qs
                    ]
                }
            return body
    raise HTTPException(status_code=404, detail="Chapter not found")


class QuizSubmission(BaseModel):
    answers: dict[str, int]  # {question_id: chosen_index}


@app.post("/novice/learn/chapters/{slug}/quiz")
def learn_quiz(slug: str, submission: QuizSubmission) -> dict[str, Any]:
    for ch in LEARN_CHAPTERS:
        if ch["slug"] == slug and ch.get("quiz"):
            qs = ch["quiz"]["questions"]
            total = len(qs)
            correct = 0
            details = []
            for q in qs:
                chosen = submission.answers.get(q["id"], None)
                is_correct = chosen == q["answer"]
                correct += int(is_correct)
                details.append(
                    {
                        "id": q["id"],
                        "correct": is_correct,
                        "explain": q.get("explain", ""),
                    }
                )
            return {"correct": correct, "total": total, "details": details}
    raise HTTPException(status_code=404, detail="Quiz not found for chapter")


@app.post("/novice/explore/simulate", response_model=SimulateResponse)
def explore_simulate(req: SimulateRequest) -> SimulateResponse:
    t, flux, expected, meta = simulate_light_curve(req)
    return SimulateResponse(time=t.tolist(), flux=flux.tolist(), expected=expected.tolist(), meta=meta)


# -------- Predict helpers ---------

def _parse_json_series(payload: PredictJSONRequest) -> tuple[np.ndarray, np.ndarray]:
    if not payload.series:
        raise HTTPException(status_code=400, detail="No series provided")
    s = payload.series[0]
    if len(s.t) != len(s.y):
        raise HTTPException(status_code=400, detail="Length mismatch between t and y")
    return np.asarray(s.t, dtype=float), np.asarray(s.y, dtype=float)


def _parse_csv_upload(file: UploadFile) -> tuple[np.ndarray, np.ndarray]:
    content = file.file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # Try pandas first
    if pd is not None:
        try:
            df = pd.read_csv(io.BytesIO(content))
            cols = [c for c in df.columns]
            # Flexible column names
            tcol = "time" if "time" in cols else ("t" if "t" in cols else None)
            ycol = "flux" if "flux" in cols else ("y" if "y" in cols else None)
            if not tcol or not ycol:
                raise ValueError("CSV must have columns time/flux or t/y")
            t = df[tcol].astype(float).to_numpy()
            y = df[ycol].astype(float).to_numpy()
            return t, y
        except Exception as e:  # fallback to csv module
            pass

    # csv module fallback
    try:
        f = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(f)
        t_list, y_list = [], []
        for row in reader:
            val_t = row.get("time", row.get("t"))
            val_y = row.get("flux", row.get("y"))
            if val_t is None or val_y is None:
                raise ValueError("CSV must have columns time/flux or t/y")
            t_list.append(float(val_t))
            y_list.append(float(val_y))
        if not t_list:
            raise ValueError("CSV appears empty")
        return np.asarray(t_list, dtype=float), np.asarray(y_list, dtype=float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")


def _build_preview(t: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    t_ds, y_ds = downsample(t, y, max_points=2000)
    return {"time": t_ds.tolist(), "flux": y_ds.tolist(), "downsampled": t_ds.size < t.size}


@app.post("/novice/predict", response_model=PredictResponse)
async def novice_predict(
    json_payload: Optional[PredictJSONRequest] = Body(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    # Accept either JSON or multipart file; not both
    if (json_payload is None) == (file is None):
        raise HTTPException(status_code=400, detail="Provide either JSON body or a CSV file upload")

    if json_payload is not None:
        t, y = _parse_json_series(json_payload)
        source = (json_payload.metadata or {}).get("source", "json")
    else:
        assert file is not None
        if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/csv", "text/plain"):
            # allow text/plain when pasted CSV
            raise HTTPException(status_code=415, detail="Unsupported file type; expected CSV")
        t, y = _parse_csv_upload(file)
        source = "upload"

    res = score_series(t, y)
    res["preview"] = _build_preview(t, y)

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "probability": res["probability"],
        "label": res["label"],
        "uncertainty_note": res["uncertainty_note"],
        "preview": res["preview"],
        "features": res["features"],
        "source": source,
    }
    rid = save_result(payload)

    return PredictResponse(
        result_id=rid,
        probability=payload["probability"],
        label=payload["label"],
        uncertainty_note=payload["uncertainty_note"],
        preview=payload["preview"],
        features=payload["features"],
    )


@app.get("/novice/predict/{result_id}", response_model=PredictResponse)
def novice_predict_get(result_id: str) -> PredictResponse:
    data = load_result(result_id)
    return PredictResponse(
        result_id=result_id,
        probability=float(data["probability"]),
        label=str(data["label"]),
        uncertainty_note=str(data["uncertainty_note"]),
        preview=data["preview"],
        features={k: float(v) for k, v in data["features"].items()},
    )


def _render_plot_png(t: np.ndarray, y: np.ndarray, meta: dict[str, Any]) -> bytes:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, y, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    ax.set_title("Light Curve Preview")
    ax.grid(True, alpha=0.3)
    # Small caption with meta
    caption = f"p={meta['probability']:.2f} • label={meta['label']}"
    ax.text(0.01, 0.02, caption, transform=ax.transAxes)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_plot_pdf(t: np.ndarray, y: np.ndarray, meta: dict[str, Any]) -> bytes:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, y, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    ax.set_title("Light Curve Preview")
    ax.grid(True, alpha=0.3)
    caption = f"p={meta['probability']:.2f} • label={meta['label']}"
    ax.text(0.01, 0.02, caption, transform=ax.transAxes)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@app.get("/novice/predict/{result_id}/download")
def novice_predict_download(result_id: str, format: Literal["png", "pdf"] = Query("png")):
    data = load_result(result_id)
    prev = data.get("preview", {})
    t = np.asarray(prev.get("time", []), dtype=float)
    y = np.asarray(prev.get("flux", []), dtype=float)
    if t.size == 0 or y.size == 0:
        raise HTTPException(status_code=422, detail="No preview data available for this result")

    meta = {"probability": float(data["probability"]), "label": str(data["label"])}

    content: bytes
    filename: str
    media_type: str
    if format == "png":
        content = _render_plot_png(t, y, meta)
        filename = f"result_{result_id}.png"
        media_type = "image/png"
    else:
        content = _render_plot_pdf(t, y, meta)
        filename = f"result_{result_id}.pdf"
        media_type = "application/pdf"

    # Save to temp file to return as FileResponse
    tmpdir = Path(".store/tmp")
    tmpdir.mkdir(parents=True, exist_ok=True)
    fpath = tmpdir / filename
    with fpath.open("wb") as f:
        f.write(content)
    return FileResponse(str(fpath), media_type=media_type, filename=filename)


# -----------------------------
# Run: uvicorn fastapi_novice_mode:app --reload
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_novice_mode:app", host="0.0.0.0", port=8000, reload=True)
