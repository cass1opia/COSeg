#!/usr/bin/env python3
"""
Konvertiert .npy Dateien in CSV-Format
"""

import numpy as np
import pandas as pd
import argparse
import os

def convert_npy_to_csv(input_file, output_file=None, delimiter=','):
    """
    Konvertiert eine .npy Datei in CSV-Format
    
    Args:
        input_file (str): Pfad zur .npy Datei
        output_file (str): Pfad zur Ausgabe-CSV-Datei (optional)
        delimiter (str): Trennzeichen für CSV (Standard: Komma)
    """
    # Lade die .npy Datei
    print(f"Lade {input_file}...")
    data = np.load(input_file)
    
    print(f"Datenform: {data.shape}")
    print(f"Daten-Typ: {data.dtype}")
    
    # Erstelle Spaltennamen basierend auf der Dimension
    if len(data.shape) == 1:
        # 1D Array
        columns = ['values']
        df = pd.DataFrame(data, columns=columns)
    elif len(data.shape) == 2:
        # 2D Array
        columns = [f'col_{i}' for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=columns)
    else:
        # Höherdimensionale Arrays - flach machen
        print(f"Warnung: {len(data.shape)}-dimensionale Daten werden abgeflacht")
        data_flat = data.flatten()
        columns = ['values']
        df = pd.DataFrame(data_flat, columns=columns)
    
    # Bestimme Ausgabedatei
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.csv"
    
    # Speichere als CSV
    print(f"Speichere als {output_file}...")
    df.to_csv(output_file, index=False, sep=delimiter)
    print(f"Konvertierung abgeschlossen: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Konvertiert .npy Dateien in CSV-Format')
    parser.add_argument('input_file', help='Pfad zur .npy Datei')
    parser.add_argument('-o', '--output', help='Ausgabedatei (optional)')
    parser.add_argument('-d', '--delimiter', default=',', help='Trennzeichen (Standard: Komma)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Fehler: Datei {args.input_file} existiert nicht!")
        return
    
    convert_npy_to_csv(args.input_file, args.output, args.delimiter)

if __name__ == "__main__":
    main() 