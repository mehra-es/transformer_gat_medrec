"""
MIMIC-III Clinical Database Demo (PhysioNet v1.4) loader and preprocessor.

Source: https://physionet.org/content/mimiciii-demo/1.4/
Open-access demo subset (100 patients). No credentials required for download.
"""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.dataset import split_patients_by_id
from src.data.preprocessing import patient_record_template

MIMIC_DEMO_ZIP_URL = (
    "https://physionet.org/static/published-projects/mimiciii-demo/"
    "mimic-iii-clinical-database-demo-1.4.zip"
)
MIMIC_DEMO_FOLDER = "mimic-iii-clinical-database-demo-1.4"


def mimic_csv_dir(raw_dir: Path) -> Path:
    return raw_dir / MIMIC_DEMO_FOLDER


def download_mimic_demo(raw_dir: Path, force: bool = False) -> Path:
    """Download and extract MIMIC-III demo CSVs from PhysioNet."""
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "mimic-iii-clinical-database-demo-1.4.zip"
    csv_dir = mimic_csv_dir(raw_dir)

    if csv_dir.exists() and (csv_dir / "ADMISSIONS.csv").exists() and not force:
        return csv_dir

    if not zip_path.exists() or force:
        print(f"Downloading MIMIC-III demo from PhysioNet...")
        urllib.request.urlretrieve(MIMIC_DEMO_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    return csv_dir


def _top_codes(series: pd.Series, top_k: int) -> List[str]:
    counts = series.dropna().astype(str).value_counts()
    return counts.head(top_k).index.tolist()


def _build_cooccurrence_ddi(
    med_matrix: np.ndarray,
    min_count: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build DDI-like edge weights from train-set medication co-prescription."""
    num_med = med_matrix.shape[1]
    co = np.zeros((num_med, num_med), dtype=np.float32)
    for row in med_matrix:
        idx = np.where(row > 0)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = idx[i], idx[j]
                co[a, b] += 1
                co[b, a] += 1
    adj = np.zeros((num_med, num_med), dtype=np.float32)
    for i in range(num_med):
        for j in range(i + 1, num_med):
            if co[i, j] >= min_count:
                w = min(1.0, co[i, j] / 10.0)
                adj[i, j] = w
                adj[j, i] = w
    rows, cols = np.where(adj > 0)
    if len(rows) == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.triu(adj, k=1),
        )
    edge_index = np.stack([rows, cols], axis=0)
    edge_weight = adj[rows, cols]
    return edge_index, edge_weight, np.triu(adj, k=1)


def build_mimic_demo_patients(
    raw_dir: Path,
    top_diag: int = 200,
    top_med: int = 300,
    top_lab_items: int = 20,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build patient visit records from MIMIC-III demo CSVs.

    Each hospital admission (hadm_id) is one visit, ordered by admittime per subject.
    Target medications = prescriptions on the patient's final admission in the timeline.
    """
    csv_dir = download_mimic_demo(raw_dir)
    admissions = pd.read_csv(
        csv_dir / "ADMISSIONS.csv",
        parse_dates=["admittime", "dischtime"],
    )
    diagnoses = pd.read_csv(csv_dir / "DIAGNOSES_ICD.csv")
    prescriptions = pd.read_csv(
        csv_dir / "PRESCRIPTIONS.csv",
        parse_dates=["startdate", "enddate"],
    )
    labevents = pd.read_csv(csv_dir / "LABEVENTS.csv", parse_dates=["charttime"])

    diagnoses["icd9_code"] = diagnoses["icd9_code"].astype(str)
    prescriptions["drug_key"] = (
        prescriptions["drug_name_generic"]
        .fillna(prescriptions["drug"])
        .astype(str)
        .str.strip()
        .str.lower()
    )
    prescriptions = prescriptions[prescriptions["drug_key"].ne("") & prescriptions["drug_key"].ne("nan")]

    # Global vocabularies (rebuilt after train split in preprocess_and_save)
    diag_vocab = _top_codes(diagnoses["icd9_code"], top_diag)
    med_vocab = _top_codes(prescriptions["drug_key"], top_med)
    lab_item_ids = labevents["itemid"].value_counts().head(top_lab_items).index.astype(int).tolist()

    diag_index = {c: i for i, c in enumerate(diag_vocab)}
    med_index = {c: i for i, c in enumerate(med_vocab)}
    lab_index = {int(i): j for j, i in enumerate(lab_item_ids)}

    admissions = admissions.sort_values(["subject_id", "admittime"])
    patients: List[Dict[str, Any]] = []

    for subject_id, adm_group in admissions.groupby("subject_id"):
        hadm_ids = adm_group["hadm_id"].tolist()
        if not hadm_ids:
            continue

        visit_diag, visit_med, visit_lab = [], [], []
        for hadm_id in hadm_ids:
            d_codes = diagnoses.loc[diagnoses["hadm_id"] == hadm_id, "icd9_code"]
            d_vec = np.zeros(len(diag_vocab), dtype=np.float32)
            for c in d_codes.unique():
                if c in diag_index:
                    d_vec[diag_index[c]] = 1.0

            rx = prescriptions.loc[prescriptions["hadm_id"] == hadm_id, "drug_key"]
            m_vec = np.zeros(len(med_vocab), dtype=np.float32)
            for c in rx.unique():
                if c in med_index:
                    m_vec[med_index[c]] = 1.0

            labs = labevents.loc[labevents["hadm_id"] == hadm_id]
            l_vec = np.zeros(len(lab_item_ids), dtype=np.float32)
            for itemid, sub in labs.groupby("itemid"):
                if int(itemid) not in lab_index:
                    continue
                vals = sub["valuenum"].dropna()
                if len(vals) > 0:
                    l_vec[lab_index[int(itemid)]] = float(vals.mean())

            visit_diag.append(d_vec)
            visit_med.append(m_vec)
            visit_lab.append(l_vec)

        target_hadm = hadm_ids[-1]
        rx_tgt = prescriptions.loc[prescriptions["hadm_id"] == target_hadm, "drug_key"]
        target = np.zeros(len(med_vocab), dtype=np.float32)
        for c in rx_tgt.unique():
            if c in med_index:
                target[med_index[c]] = 1.0

        if target.sum() == 0:
            continue

        patients.append(
            patient_record_template(
                patient_id=str(subject_id),
                diagnoses=np.stack(visit_diag),
                medications=np.stack(visit_med),
                labs=np.stack(visit_lab),
                target=target,
            )
        )

    meta = {
        "source": "mimiciii-demo-1.4",
        "source_url": "https://physionet.org/content/mimiciii-demo/1.4/",
        "num_diag": len(diag_vocab),
        "num_med": len(med_vocab),
        "num_lab": len(lab_item_ids),
        "num_patients": len(patients),
        "diag_vocab": diag_vocab,
        "med_vocab": med_vocab,
        "lab_itemids": lab_item_ids,
        "seed": seed,
    }
    return patients, meta


def preprocess_and_save(
    raw_dir: Path,
    processed_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    top_diag: int = 200,
    top_med: int = 300,
    top_lab_items: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Download, preprocess, split by patient, and save artifacts."""
    patients, meta = build_mimic_demo_patients(
        raw_dir, top_diag=top_diag, top_med=top_med, top_lab_items=top_lab_items, seed=seed
    )
    train_p, val_p, test_p = split_patients_by_id(
        patients, train_ratio, val_ratio, test_ratio, seed=seed
    )

    # DDI graph from train-set co-prescription on target meds
    train_targets = np.stack([p["target"] for p in train_p], axis=0)
    edge_index, edge_weight, adj_upper = _build_cooccurrence_ddi(train_targets)

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train_p), ("val", val_p), ("test", test_p)]:
        with (processed_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(split, f)

    np.savez(
        processed_dir / "ddi_graph.npz",
        edge_index=edge_index,
        edge_weight=edge_weight,
        adj_upper=adj_upper,
    )
    meta.update(
        {
            "train_size": len(train_p),
            "val_size": len(val_p),
            "test_size": len(test_p),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }
    )
    with (processed_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def load_mimic_demo_splits(
    processed_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load preprocessed MIMIC demo splits and metadata."""
    with (processed_dir / "train.pkl").open("rb") as f:
        train_p = pickle.load(f)
    with (processed_dir / "val.pkl").open("rb") as f:
        val_p = pickle.load(f)
    with (processed_dir / "test.pkl").open("rb") as f:
        test_p = pickle.load(f)
    with (processed_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return train_p, val_p, test_p, meta


def load_mimic_demo_ddi(processed_dir: Path):
    """Load DDI graph arrays saved during preprocessing."""
    import torch

    data = np.load(processed_dir / "ddi_graph.npz")
    return (
        torch.tensor(data["edge_index"], dtype=torch.long),
        torch.tensor(data["edge_weight"], dtype=torch.float32),
        torch.tensor(data["adj_upper"], dtype=torch.float32),
    )
