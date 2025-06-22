#!/usr/bin/env python3
"""
Konvertiert .npy Dateien in TXT-Format
"""

import numpy as np
import argparse
import os

def convert_npy_to_txt(input_file, output_file=None, delimiter=' ', precision=6):
    """
    Konvertiert eine .npy Datei in TXT-Format
    
    Args:
        input_file (str): Pfad zur .npy Datei
        output_file (str): Pfad zur Ausgabe-TXT-Datei (optional)
        delimiter (str): Trennzeichen zwischen Werten (Standard: Leerzeichen)
        precision (int): Anzahl der Dezimalstellen (Standard: 6)
    """
    # Lade die .npy Datei
    print(f"Lade {input_file}...")
    data = np.load(input_file)
    
    print(f"Datenform: {data.shape}")
    print(f"Daten-Typ: {data.dtype}")
    
    # Bestimme Ausgabedatei
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.txt"
    
    # Speichere als TXT
    print(f"Speichere als {output_file}...")
    
    # Verwende numpy.savetxt für einfache Konvertierung
    np.savetxt(output_file, data, delimiter=delimiter, fmt=f'%.{precision}f')
    
    print(f"Konvertierung abgeschlossen: {output_file}")
    
    return output_file

def convert_npy_to_txt_manual(input_file, output_file=None, delimiter=' '):
    """
    Alternative manuelle Konvertierung mit mehr Kontrolle
    """
    # Lade die .npy Datei
    print(f"Lade {input_file}...")
    data = np.load(input_file)
    
    print(f"Datenform: {data.shape}")
    print(f"Daten-Typ: {data.dtype}")
    
    # Bestimme Ausgabedatei
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.txt"
    
    # Speichere als TXT
    print(f"Speichere als {output_file}...")
    
    with open(output_file, 'w') as f:
        if len(data.shape) == 1:
            # 1D Array
            for value in data:
                f.write(f"{value}\n")
        elif len(data.shape) == 2:
            # 2D Array
            for row in data:
                f.write(delimiter.join(map(str, row)) + '\n')
        else:
            # Höherdimensionale Arrays
            data_flat = data.flatten()
            for value in data_flat:
                f.write(f"{value}\n")
    
    print(f"Konvertierung abgeschlossen: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Konvertiert .npy Dateien in TXT-Format')
    parser.add_argument('input_file', help='Pfad zur .npy Datei')
    parser.add_argument('-o', '--output', help='Ausgabedatei (optional)')
    parser.add_argument('-d', '--delimiter', default=' ', help='Trennzeichen (Standard: Leerzeichen)')
    parser.add_argument('-p', '--precision', type=int, default=6, help='Dezimalstellen (Standard: 6)')
    parser.add_argument('--manual', action='store_true', help='Verwende manuelle Konvertierung')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Fehler: Datei {args.input_file} existiert nicht!")
        return
    
    if args.manual:
        convert_npy_to_txt_manual(args.input_file, args.output, args.delimiter)
    else:
        convert_npy_to_txt(args.input_file, args.output, args.delimiter, args.precision)

if __name__ == "__main__":
    main() 