#!/usr/bin/env python3
"""
Einfache Python-Beispiele zur Konvertierung von .npy Dateien
"""

import numpy as np
import pandas as pd

# ============================================================================
# BEISPIEL 1: Einfache Konvertierung zu CSV
# ============================================================================

def example_csv_conversion():
    """Einfache Konvertierung zu CSV"""
    print("=== CSV Konvertierung ===")
    
    # Lade .npy Datei
    data = np.load('deine_datei.npy')  # Ersetze mit deinem Dateipfad
    
    # Konvertiere zu DataFrame und speichere als CSV
    df = pd.DataFrame(data)
    df.to_csv('output.csv', index=False)
    print("CSV gespeichert als 'output.csv'")

# ============================================================================
# BEISPIEL 2: Einfache Konvertierung zu TXT
# ============================================================================

def example_txt_conversion():
    """Einfache Konvertierung zu TXT"""
    print("=== TXT Konvertierung ===")
    
    # Lade .npy Datei
    data = np.load('deine_datei.npy')  # Ersetze mit deinem Dateipfad
    
    # Speichere als TXT mit numpy.savetxt
    np.savetxt('output.txt', data, delimiter=' ', fmt='%.6f')
    print("TXT gespeichert als 'output.txt'")

# ============================================================================
# BEISPIEL 3: Konvertierung mit benutzerdefinierten Spaltennamen
# ============================================================================

def example_with_column_names():
    """Konvertierung mit benutzerdefinierten Spaltennamen"""
    print("=== Konvertierung mit Spaltennamen ===")
    
    # Lade .npy Datei
    data = np.load('deine_datei.npy')  # Ersetze mit deinem Dateipfad
    
    # Definiere Spaltennamen (anpassen an deine Daten)
    if len(data.shape) == 2:
        columns = ['x', 'y', 'z', 'r', 'g', 'b', 'label']  # Für Point Cloud Daten
        df = pd.DataFrame(data, columns=columns)
    else:
        df = pd.DataFrame(data)
    
    # Speichere als CSV
    df.to_csv('output_with_columns.csv', index=False)
    print("CSV mit Spaltennamen gespeichert als 'output_with_columns.csv'")

# ============================================================================
# BEISPIEL 4: Batch-Konvertierung mehrerer .npy Dateien
# ============================================================================

def batch_conversion():
    """Konvertiert alle .npy Dateien in einem Verzeichnis"""
    print("=== Batch-Konvertierung ===")
    
    import glob
    import os
    
    # Finde alle .npy Dateien
    npy_files = glob.glob('*.npy')  # Alle .npy Dateien im aktuellen Verzeichnis
    
    for npy_file in npy_files:
        print(f"Konvertiere {npy_file}...")
        
        # Lade Daten
        data = np.load(npy_file)
        
        # Erstelle Basisnamen
        base_name = os.path.splitext(npy_file)[0]
        
        # Konvertiere zu CSV
        df = pd.DataFrame(data)
        csv_file = f"{base_name}.csv"
        df.to_csv(csv_file, index=False)
        
        # Konvertiere zu TXT
        txt_file = f"{base_name}.txt"
        np.savetxt(txt_file, data, delimiter=' ', fmt='%.6f')
        
        print(f"  → {csv_file} und {txt_file} erstellt")

# ============================================================================
# BEISPIEL 5: Einzeilige Konvertierungen (für schnelle Verwendung)
# ============================================================================

def one_liner_examples():
    """Einzeilige Konvertierungen für schnelle Verwendung"""
    print("=== Einzeilige Konvertierungen ===")
    
    # CSV Konvertierung in einer Zeile:
    # pd.DataFrame(np.load('datei.npy')).to_csv('output.csv', index=False)
    
    # TXT Konvertierung in einer Zeile:
    # np.savetxt('output.txt', np.load('datei.npy'), delimiter=' ', fmt='%.6f')
    
    print("Einzeilige Befehle (auskommentiert):")
    print("# pd.DataFrame(np.load('datei.npy')).to_csv('output.csv', index=False)")
    print("# np.savetxt('output.txt', np.load('datei.npy'), delimiter=' ', fmt='%.6f')")

# ============================================================================
# HAUPTFUNKTION
# ============================================================================

if __name__ == "__main__":
    print("Beispiele für .npy Konvertierung")
    print("=" * 40)
    
    # Zeige Beispiele (ohne Ausführung)
    example_csv_conversion.__doc__
    example_txt_conversion.__doc__
    example_with_column_names.__doc__
    batch_conversion.__doc__
    one_liner_examples.__doc__
    
    print("\nVerwende diese Funktionen oder kopiere den Code in dein Skript.")
    print("Vergiss nicht, 'deine_datei.npy' durch deinen tatsächlichen Dateipfad zu ersetzen!") 