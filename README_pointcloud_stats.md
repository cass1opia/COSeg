# Punktwolken-Statistik Generator

Dieses Skript analysiert einen Ordner mit Punktwolken-Dateien (.npy) und erstellt eine CSV-Datei mit detaillierten Statistiken.

## Funktionalität

Das Skript generiert für jede Punktwolke folgende Informationen:
- **id**: Eindeutige ID (fortlaufend)
- **name**: Dateiname ohne Erweiterung
- **gesamtpunktanzahl**: Gesamtzahl der Punkte in der Punktwolke
- **streetsign_amount**: Anzahl der Punkte mit Label ungleich 0 (Straßenschilder)
- **min_x, max_x**: Minimale und maximale X-Koordinaten
- **min_y, max_y**: Minimale und maximale Y-Koordinaten
- **min_z, max_z**: Minimale und maximale Z-Koordinaten

## Voraussetzungen

```bash
pip install numpy pandas
```

## Verwendung

### Grundlegende Verwendung

```bash
python generate_pointcloud_stats.py /pfad/zu/punktwolken_ordner
```

### Mit benutzerdefiniertem Ausgabedateinamen

```bash
python generate_pointcloud_stats.py /pfad/zu/punktwolken_ordner -o meine_statistiken.csv
```

### Ausführliche Ausgabe

```bash
python generate_pointcloud_stats.py /pfad/zu/punktwolken_ordner -v
```

## Kommandozeilen-Optionen

- `input_folder`: Pfad zum Ordner mit .npy Dateien (erforderlich)
- `-o, --output`: Ausgabe-CSV-Datei (Standard: `pointcloud_statistics.csv`)
- `-v, --verbose`: Ausführliche Logging-Ausgabe

## Datenformat

Das Skript erwartet .npy Dateien mit folgender Struktur:
- **Spalte 0-2**: X, Y, Z Koordinaten (erforderlich)
- **Spalte 3+**: Labels (optional, Standard: alle 0)

## Beispiel-Ausgabe

```csv
id,name,gesamtpunktanzahl,streetsign_amount,min_x,max_x,min_y,max_y,min_z,max_z
1,pointcloud_001,15420,1250,-12.456,45.789,-8.234,32.567,0.123,15.678
2,pointcloud_002,20340,1890,-15.123,42.456,-12.789,28.234,-2.456,18.901
```

## Fehlerbehandlung

Das Skript behandelt folgende Fälle robust:
- Ungültige .npy Dateien werden übersprungen
- Fehlende Label-Spalten werden mit Nullen gefüllt
- Unerwartete Datenformate werden protokolliert
- Fehler beim Laden einzelner Dateien stoppen nicht die gesamte Verarbeitung

## Logging

Das Skript protokolliert alle wichtigen Schritte und Warnungen. Bei Verwendung der `-v` Option werden zusätzliche Debug-Informationen ausgegeben.

## Ausgabe

Nach erfolgreicher Verarbeitung wird eine Zusammenfassung mit folgenden Informationen ausgegeben:
- Anzahl der verarbeiteten Dateien
- Gesamtanzahl aller Punkte
- Gesamtanzahl der Straßenschilder
- Globale Bounding Box aller Punktwolken
