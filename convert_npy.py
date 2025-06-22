#!/usr/bin/env python3
"""
Universelles Skript zur Konvertierung von .npy Dateien in verschiedene Formate
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys

def load_npy_file(file_path):
    """Lädt eine .npy Datei und gibt Informationen aus"""
    try:
        data = np.load(file_path)
        print(f"✓ Datei geladen: {file_path}")
        print(f"  Form: {data.shape}")
        print(f"  Typ: {data.dtype}")
        print(f"  Min: {np.min(data)}")
        print(f"  Max: {np.max(data)}")
        return data
    except Exception as e:
        print(f"✗ Fehler beim Laden von {file_path}: {e}")
        return None

def convert_to_csv(data, output_file, delimiter=','):
    """Konvertiert NumPy-Array zu CSV"""
    try:
        if len(data.shape) == 1:
            columns = ['values']
            df = pd.DataFrame(data, columns=columns)
        elif len(data.shape) == 2:
            columns = [f'col_{i}' for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=columns)
        else:
            print(f"Warnung: {len(data.shape)}-dimensionale Daten werden abgeflacht")
            data_flat = data.flatten()
            columns = ['values']
            df = pd.DataFrame(data_flat, columns=columns)
        
        df.to_csv(output_file, index=False, sep=delimiter)
        print(f"✓ CSV gespeichert: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Fehler beim Speichern als CSV: {e}")
        return False

def convert_to_txt(data, output_file, delimiter=' ', precision=6):
    """Konvertiert NumPy-Array zu TXT"""
    try:
        np.savetxt(output_file, data, delimiter=delimiter, fmt=f'%.{precision}f')
        print(f"✓ TXT gespeichert: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Fehler beim Speichern als TXT: {e}")
        return False

def convert_to_npz(data, output_file):
    """Konvertiert NumPy-Array zu NPZ (komprimiert)"""
    try:
        np.savez_compressed(output_file, data=data)
        print(f"✓ NPZ gespeichert: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Fehler beim Speichern als NPZ: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Konvertiert .npy Dateien in verschiedene Formate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python convert_npy.py data.npy --csv
  python convert_npy.py data.npy --txt --delimiter "\\t"
  python convert_npy.py data.npy --csv --txt --output "output"
  python convert_npy.py data.npy --npz
        """
    )
    
    parser.add_argument('input_file', help='Pfad zur .npy Datei')
    parser.add_argument('-o', '--output', help='Basisname für Ausgabedateien (ohne Erweiterung)')
    parser.add_argument('--csv', action='store_true', help='Konvertiere zu CSV')
    parser.add_argument('--txt', action='store_true', help='Konvertiere zu TXT')
    parser.add_argument('--npz', action='store_true', help='Konvertiere zu NPZ (komprimiert)')
    parser.add_argument('-d', '--delimiter', default=',', help='Trennzeichen für CSV/TXT (Standard: Komma)')
    parser.add_argument('-p', '--precision', type=int, default=6, help='Dezimalstellen für TXT (Standard: 6)')
    parser.add_argument('--info', action='store_true', help='Zeige nur Informationen über die Datei')
    
    args = parser.parse_args()
    
    # Prüfe ob Eingabedatei existiert
    if not os.path.exists(args.input_file):
        print(f"✗ Fehler: Datei {args.input_file} existiert nicht!")
        sys.exit(1)
    
    # Lade Daten
    data = load_npy_file(args.input_file)
    if data is None:
        sys.exit(1)
    
    # Nur Informationen anzeigen
    if args.info:
        print("\nDatenanalyse:")
        print(f"  Anzahl Elemente: {data.size}")
        print(f"  Speichergröße: {data.nbytes / 1024:.2f} KB")
        if len(data.shape) > 1:
            print(f"  Dimensionen: {data.shape}")
        return
    
    # Bestimme Ausgabebasisname
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
    else:
        base_name = args.output
    
    # Führe Konvertierungen durch
    success_count = 0
    
    if args.csv:
        output_file = f"{base_name}.csv"
        if convert_to_csv(data, output_file, args.delimiter):
            success_count += 1
    
    if args.txt:
        output_file = f"{base_name}.txt"
        if convert_to_txt(data, output_file, args.delimiter, args.precision):
            success_count += 1
    
    if args.npz:
        output_file = f"{base_name}.npz"
        if convert_to_npz(data, output_file):
            success_count += 1
    
    # Wenn keine Konvertierung angegeben wurde, zeige Hilfe
    if not any([args.csv, args.txt, args.npz]):
        print("\nKeine Konvertierung angegeben. Verwende --csv, --txt oder --npz")
        print("Verwende --help für weitere Informationen")
        return
    
    print(f"\n✓ {success_count} Konvertierung(en) erfolgreich abgeschlossen!")

if __name__ == "__main__":
    main() 