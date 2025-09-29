#!/usr/bin/env python3
"""
Analysiert alle .npy-Blockdateien und erzeugt eine CSV mit folgenden Spalten:
- name: Blockname (Dateiname ohne Endung)
- je eine Spalte pro Klasse (1..13 nach Mapping), Werte = Punktanzahl der Klasse
- label: Klassenname mit den meisten Punkten im Block (leer, falls keine der Klassen 1..13 vorkommt)
- run: aus dem Blocknamen extrahierter Run-Wert (z.B. "Run10_1..." -> 10.1, "Run6__..." -> 6)

Standard-Eingabeordner:
  /sc/projects/sci-doellner/chair/adrian.schmidt/coseg_data/essen-road/processed/blocks_bs5.0_s5.0/data

Ausgabe:
  stats/blocks_class_counts.csv
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Klassen-Mapping laut Nutzerangabe (IDs 0..13)
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "Andere", #no use
    1: "StreetSign",
    2: "TrafficSign", #no use
    3: "CircularTrafficSign", 
    4: "OctagonalTrafficSign",
    5: "RectangularTrafficSign",
    6: "TriangularTrafficSign",
    7: "DirectionSign",
    8: "FlippedTriangularTrafficSign",
    9: "PriorityRoad",
    10: "OneWayStreet", # no use
    11: "Zone",
    12: "Exit", #no use
    13: "DistanceMarker",
}

CLASS_IDS: List[int] = list(CLASS_ID_TO_NAME.keys())
CLASS_NAMES: List[str] = [CLASS_ID_TO_NAME[i] for i in CLASS_IDS]


def extract_run_value(block_name: str) -> float:
    """
    Extrahiert den Run-Wert aus dem Blocknamen.
    Regeln:
      - "Run10_1..." -> 10.1
      - "Run6__..." -> 6
      - Allgemein: Nach "Run" erste Ganzzahl = Hauptindex; wenn danach "_" und weitere Zahl, als Dezimalteil.
    """
    # Suche nach Run gefolgt von Ziffern und optional _Ziffern
    m = re.search(r"Run(\d+)(?:_+(\d+))?", block_name)
    if not m:
        return np.nan
    major = int(m.group(1))
    minor_str = m.group(2)
    if minor_str is None or len(minor_str) == 0:
        return float(major)
    # Zusammensetzen als Dezimalwert: 10 und 1 -> 10.1 ; 12 und 34 -> 12.34
    try:
        minor = int(minor_str)
        # Stelle die Dezimalzahl korrekt her abhängig von Ziffernanzahl
        scale = 10 ** len(minor_str)
        return major + minor / scale
    except ValueError:
        return float(major)


def count_labels_in_block(npy_path: Path) -> Tuple[str, Dict[str, int], str, float, int, int]:
    """
    Zählt pro Klasse (1..13) die Punkte im Block.

    Returns:
        name (str): Blockname ohne .npy
        counts_by_class_name (dict): {Klassenname: Anzahl}
        dominant_label_name (str): Klassenname mit max Punkten ("" falls keine der Klassen vorhanden)
        run_value (float): extrahierter Run-Wert
        total_points (int): Gesamtanzahl Punkte im Block
        label_id (int): ID der dominanten Klasse (0 falls keine der Klassen vorhanden)
    """
    data = np.load(str(npy_path))
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Unerwartete Datenform in {npy_path}: {data.shape}")

    labels = data[:, 6].astype(int)

    # Zähle nur die Klassen aus dem Mapping (inkl. 0="Andere")
    counts_by_id = {cid: int(np.count_nonzero(labels == cid)) for cid in CLASS_IDS}
    counts_by_class_name = {CLASS_ID_TO_NAME[cid]: counts_by_id[cid] for cid in CLASS_IDS}

    # Dominante Klasse bestimmen
    if sum(counts_by_id.values()) == 0:
        dominant_label_name = ""
        dominant_id = 0
    else:
        dominant_id = max((k for k in counts_by_id if k != 0), key=lambda k: counts_by_id[k], default=0)
        dominant_label_name = CLASS_ID_TO_NAME[dominant_id]

    name = npy_path.stem
    run_value = extract_run_value(name)
    total_points = int(data.shape[0])
    return name, counts_by_class_name, dominant_label_name, run_value, total_points, dominant_id


def main():
    parser = argparse.ArgumentParser(description="Analysiert .npy-Blöcke und erzeugt Klassen-Statistiken als CSV")
    parser.add_argument(
        "--input_dir",
        default="/sc/projects/sci-doellner/chair/adrian.schmidt/coseg_data/essen-road/processed/blocks_bs5.0_s5.0/data",
        help="Ordner mit .npy Block-Dateien"
    )
    parser.add_argument(
        "--output_csv",
        default="stats/blocks_class_counts.csv",
        help="Pfad zur Ausgabedatei (CSV)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Eingabeordner nicht gefunden oder kein Ordner: {input_dir}")

    npy_files = sorted(input_dir.glob("*.npy"))
    if len(npy_files) == 0:
        raise FileNotFoundError(f"Keine .npy-Dateien in {input_dir} gefunden")

    rows: List[Dict[str, object]] = []
    for f in npy_files:
        name, counts_by_class_name, dominant_label_name, run_value, total_points, label_id = count_labels_in_block(f)
        row = {"name": name, "run": run_value, "total": total_points, "label": dominant_label_name, "label_id": label_id, **counts_by_class_name}
        rows.append(row)

    # DataFrame mit definierter Spaltenreihenfolge: name, run, total, label, label_id, Klassenspalten
    df = pd.DataFrame(rows)
    column_order = ["name", "run", "total", "label", "label_id"] + CLASS_NAMES
    # Fehlende Spalten ergänzen, falls einige Klassen nicht vorkamen
    for cn in CLASS_NAMES:
        if cn not in df.columns:
            df[cn] = 0
    df = df[column_order]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"CSV gespeichert: {out_path} | Blöcke: {len(df)} | Klassen: {len(CLASS_NAMES)}")


if __name__ == "__main__":
    main()


