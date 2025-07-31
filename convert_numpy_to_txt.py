#!/usr/bin/env python3
"""
Skript zur Konvertierung von NumPy-Arrays in TXT-Dateien mit Header.
Header: x y z r g b class intensity
"""

import numpy as np
import argparse
import os
import sys


def convert_numpy_to_txt(input_file, output_file=None):
    """
    Konvertiert ein NumPy-Array in eine TXT-Datei mit Header.
    
    Args:
        input_file (str): Pfad zur .npy Datei
        output_file (str): Pfad zur Ausgabe-TXT-Datei (optional)
    """
    
    # NumPy-Array laden
    try:
        data = np.load(input_file)
        print(f"Geladenes Array hat die Form: {data.shape}")
        print(f"Datentyp: {data.dtype}")
    except Exception as e:
        print(f"Fehler beim Laden der Datei {input_file}: {e}")
        return
    
    # Ausgabedateiname bestimmen
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.txt"
    
    # Header definieren
    header = "x y z r g b class intensity"
    
    # Prüfen, ob das Array die richtige Anzahl von Spalten hat
    if len(data.shape) == 1:
        # Wenn es ein 1D-Array ist, versuchen wir es zu reshapen
        if data.size % 8 == 0:
            data = data.reshape(-1, 8)
        else:
            print(f"Warnung: Array hat {data.size} Elemente, aber 8 Spalten erwartet")
            print("Versuche mit verfügbaren Spalten zu arbeiten...")
    elif len(data.shape) == 2:
        if data.shape[1] != 8:
            print(f"Warnung: Array hat {data.shape[1]} Spalten, aber 8 erwartet")
            print("Verwende verfügbare Spalten...")
    else:
        print(f"Unerwartete Array-Form: {data.shape}")
        return
    
    # Daten in TXT-Datei schreiben
    try:
        with open(output_file, 'w') as f:
            # Header schreiben
            f.write(header + '\n')
            
            # Daten schreiben
            for row in data:
                # Sicherstellen, dass wir 8 Werte haben
                if len(row) >= 8:
                    line = ' '.join([str(val) for val in row[:8]])
                else:
                    # Falls weniger als 8 Spalten, mit Nullen auffüllen
                    line = ' '.join([str(val) for val in row])
                    line += ' 0' * (8 - len(row))
                
                f.write(line + '\n')
        
        print(f"Konvertierung erfolgreich! Ausgabedatei: {output_file}")
        print(f"Anzahl Zeilen geschrieben: {len(data)}")
        
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Konvertiert NumPy-Arrays in TXT-Dateien mit Header')
    parser.add_argument('input_file', help='Pfad zur .npy Datei')
    parser.add_argument('-o', '--output', help='Ausgabedatei (optional)')
    
    args = parser.parse_args()
    
    # Prüfen, ob Eingabedatei existiert
    if not os.path.exists(args.input_file):
        print(f"Fehler: Datei {args.input_file} existiert nicht!")
        sys.exit(1)
    
    # Konvertierung durchführen
    convert_numpy_to_txt(args.input_file, args.output)


if __name__ == "__main__":
    main() 