#!/usr/bin/env python3
"""
Skript zur Generierung von Statistiken für Punktwolken.
Erstellt eine CSV-Datei mit ID, Name, Gesamtpunktanzahl, Anzahl Straßenschilder,
und Bounding Box Koordinaten für jede Punktwolke.
"""

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_pointcloud(file_path):
    """
    Analysiert eine einzelne Punktwolke und gibt Statistiken zurück.
    
    Args:
        file_path (str): Pfad zur .npy Datei
        
    Returns:
        dict: Dictionary mit den Statistiken der Punktwolke
    """
    try:
        # Punktwolke laden
        pointcloud = np.load(file_path)
        
        # Überprüfen der Datenstruktur
        if pointcloud.ndim != 2:
            logger.warning(f"Unerwartete Dimensionen in {file_path}: {pointcloud.shape}")
            return None
            
        # Mindestens 3 Spalten für x, y, z Koordinaten
        if pointcloud.shape[1] < 3:
            logger.warning(f"Zu wenige Spalten in {file_path}: {pointcloud.shape[1]}")
            return None
            
                
        xyz = pointcloud[:, :3]
        xyz_min = np.amin(xyz, axis=0)
        xyz -= xyz_min
        xyz_max = np.amax(xyz, axis=0)
        # Labels extrahieren (falls vorhanden, sonst alle 0)

        print(pointcloud[:2])
        if pointcloud.shape[1] >= 4:
            labels = pointcloud[:, 3]
        else:
            labels = np.zeros(pointcloud.shape[0])
        
        # Statistiken berechnen
        stats = {
            'gesamtpunktanzahl': len(pointcloud),
            'streetsign_amount': np.sum(labels != 0),
            'min_x': xyz_min[0],
            'max_x': xyz_max[0],
            'min_y': xyz_min[1],
            'max_y': xyz_max[1],
            'min_z': xyz_min[2],
            'max_z': xyz_max[2]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Fehler beim Laden von {file_path}: {e}")
        return None

def process_pointcloud_folder(folder_path, output_csv):
    """
    Verarbeitet alle .npy Dateien in einem Ordner und erstellt eine CSV-Datei.
    
    Args:
        folder_path (str): Pfad zum Ordner mit .npy Dateien
        output_csv (str): Pfad zur Ausgabe-CSV-Datei
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        logger.error(f"Ordner {folder_path} existiert nicht!")
        return
    
    if not folder_path.is_dir():
        logger.error(f"{folder_path} ist kein Ordner!")
        return
    
    # Alle .npy Dateien finden
    npy_files = list(folder_path.glob("*.npy"))
    
    if not npy_files:
        logger.warning(f"Keine .npy Dateien in {folder_path} gefunden!")
        return
    
    logger.info(f"Gefunden: {len(npy_files)} .npy Dateien")
    
    # Liste für alle Statistiken
    all_stats = []
    
    # Jede Datei verarbeiten
    for i, file_path in enumerate(sorted(npy_files)):
        logger.info(f"Verarbeite {i+1}/{len(npy_files)}: {file_path.name}")
        
        # Statistiken berechnen
        stats = analyze_pointcloud(str(file_path))
        
        if stats is not None:
            # Datei-Informationen hinzufügen
            stats['id'] = i + 1
            stats['name'] = file_path.stem  # Dateiname ohne Erweiterung
            all_stats.append(stats)
        else:
            logger.warning(f"Konnte {file_path.name} nicht verarbeiten")
    
    if not all_stats:
        logger.error("Keine gültigen Punktwolken gefunden!")
        return
    
    # DataFrame erstellen
    df = pd.DataFrame(all_stats)
    
    # Spalten in gewünschter Reihenfolge anordnen
    column_order = ['id', 'name', 'gesamtpunktanzahl', 'streetsign_amount', 
                    'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']
    df = df[column_order]
    
    # CSV speichern
    df.to_csv(output_csv, index=False, float_format='%.6f')
    
    logger.info(f"CSV-Datei erfolgreich erstellt: {output_csv}")
    logger.info(f"Verarbeitete Punktwolken: {len(df)}")
    
    # Zusammenfassung ausgeben
    print("\n=== ZUSAMMENFASSUNG ===")
    print(f"Verarbeitete Dateien: {len(df)}")
    print(f"Ausgabe-CSV: {output_csv}")
    print(f"Gesamtpunkte: {df['gesamtpunktanzahl'].sum():,}")
    print(f"Straßenschilder gesamt: {df['streetsign_amount'].sum():,}")
    print(f"X-Bereich: {df['min_x'].min():.3f} bis {df['max_x'].max():.3f}")
    print(f"Y-Bereich: {df['min_y'].min():.3f} bis {df['max_y'].max():.3f}")
    print(f"Z-Bereich: {df['min_z'].min():.3f} bis {df['max_z'].max():.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="Generiert Statistiken für Punktwolken aus .npy Dateien"
    )
    parser.add_argument(
        "--input_folder",
        help="Ordner mit .npy Punktwolken-Dateien",
        default="/sc/projects/sci-doellner/chair/adrian.schmidt/coseg_data/essen-road/processed"
    )
    parser.add_argument(
        "-o", "--output",
        default="stats/pointcloud_statistics.csv",
        help="Ausgabe-CSV-Datei (Standard: pointcloud_statistics.csv)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Ausführliche Ausgabe"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verarbeitung starten
    process_pointcloud_folder(args.input_folder, args.output)

if __name__ == "__main__":
    main()
