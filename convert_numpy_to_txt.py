#!/usr/bin/env python3
"""
Skript zur Konvertierung von NumPy-Arrays in TXT-Dateien mit Header.
Konvertiert alle .npy Dateien mit einem bestimmten Prefix in einem Ordner.
Header: x y z r g b class intensity
"""

import numpy as np
import argparse
import os
import sys
import glob


def convert_numpy_to_txt(input_file, output_file):
    """
    Konvertiert ein NumPy-Array in eine TXT-Datei mit Header.
    
    Args:
        input_file (str): Pfad zur .npy Datei
        output_file (str): Pfad zur Ausgabe-TXT-Datei
    
    Returns:
        bool: True wenn erfolgreich, False bei Fehler
    """
    
    # NumPy-Array laden
    try:
        data = np.load(input_file)
        print(f"Geladenes Array hat die Form: {data.shape}")
        print(f"Datentyp: {data.dtype}")
    except Exception as e:
        print(f"Fehler beim Laden der Datei {input_file}: {e}")
        return False
    
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
        return False
    
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
        return True
        
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei {output_file}: {e}")
        return False


def convert_folder_with_prefix(input_folder, prefix, output_folder):
    """
    Konvertiert alle .npy Dateien mit einem bestimmten Prefix in einem Ordner.
    
    Args:
        input_folder (str): Eingabeordner mit .npy Dateien
        prefix (str): Prefix für die zu konvertierenden Dateien
        output_folder (str): Ausgabeordner für die .txt Dateien
    """
    
    # Prüfen, ob Eingabeordner existiert
    if not os.path.exists(input_folder):
        print(f"Fehler: Eingabeordner {input_folder} existiert nicht!")
        return False
    
    # Ausgabeordner erstellen, falls er nicht existiert
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Ausgabeordner {output_folder} erstellt.")
        except Exception as e:
            print(f"Fehler beim Erstellen des Ausgabeordners {output_folder}: {e}")
            return False
    
    # Alle .npy Dateien mit dem Prefix finden
    pattern = os.path.join(input_folder, f"{prefix}*.npy")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"Keine .npy Dateien mit Prefix '{prefix}' in {input_folder} gefunden.")
        print(f"Gesuchtes Pattern: {pattern}")
        return False
    
    print(f"Gefundene Dateien mit Prefix '{prefix}': {len(matching_files)}")
    
    # Dateien konvertieren
    successful_conversions = 0
    failed_conversions = 0
    
    for input_file in matching_files:
        # Ausgabedateiname bestimmen
        filename = os.path.basename(input_file)
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_folder, f"{base_name}.txt")
        
        print(f"\nKonvertiere: {input_file} -> {output_file}")
        
        if convert_numpy_to_txt(input_file, output_file):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    # Zusammenfassung ausgeben
    print(f"\n=== Konvertierung abgeschlossen ===")
    print(f"Erfolgreiche Konvertierungen: {successful_conversions}")
    print(f"Fehlgeschlagene Konvertierungen: {failed_conversions}")
    print(f"Gesamt: {len(matching_files)}")
    
    return failed_conversions == 0


def main():
    parser = argparse.ArgumentParser(description='Konvertiert NumPy-Arrays in TXT-Dateien mit Header')
    parser.add_argument('input_folder', help='Eingabeordner mit .npy Dateien')
    parser.add_argument('prefix', help='Prefix für die zu konvertierenden Dateien')
    parser.add_argument('-o', '--output', help='Ausgabeordner (optional, Standard: input_folder_converted)')
    
    args = parser.parse_args()
    
    # Prüfen, ob Eingabeordner existiert
    if not os.path.exists(args.input_folder):
        print(f"Fehler: Eingabeordner {args.input_folder} existiert nicht!")
        sys.exit(1)
    
    # Ausgabeordner bestimmen
    if args.output is None:
        output_folder = f"{args.input_folder}_converted"
    else:
        output_folder = args.output
    
    # Konvertierung durchführen
    success = convert_folder_with_prefix(args.input_folder, args.prefix, output_folder)
    
    if success:
        print("Alle Konvertierungen erfolgreich abgeschlossen!")
        sys.exit(0)
    else:
        print("Einige Konvertierungen sind fehlgeschlagen!")
        sys.exit(1)


if __name__ == "__main__":
    main() 