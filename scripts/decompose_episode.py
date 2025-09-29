#!/usr/bin/env python3
"""
Skript zum Zerlegen einer Episode in ihre Bestandteile und Speichern als Punktwolken-Dateien.
Verwendung: python decompose_episode.py <episode_file.pt> [output_dir]
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def load_episode(episode_file):
    """Lädt eine Episode aus einer .pt Datei."""
    if not os.path.exists(episode_file):
        raise FileNotFoundError(f"Episode-Datei nicht gefunden: {episode_file}")
    
    data_file = torch.load(episode_file)
    return {
        'support_feat': data_file["support_feat"],
        'support_label': data_file["support_label"], 
        'query_feat': data_file["query_feat"],
        'query_label': data_file["query_label"],
        'sampled_classes': data_file["sampled_classes"]
    }

def save_pointcloud_as_txt(points, labels, filename, class_names=None):
    """
    Speichert Punktwolke als .txt Datei im Format: x y z r g b label [class_name]
    
    Args:
        points: Tensor/Array mit Shape (N, 6) - [x, y, z, r, g, b]
        labels: Tensor/Array mit Shape (N,) - Labels
        filename: Ausgabedatei
        class_names: Optional dict {class_id: class_name}
    """
    points_np = points.cpu().numpy() if torch.is_tensor(points) else points
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    # Erstelle Ausgabedatei
    with open(filename, 'w') as f:
        f.write("# Punktwolke: x y z r g b label")
        if class_names:
            f.write(" class_name")
        f.write("\n")
        
        for i in range(len(points_np)):
            x, y, z, r, g, b = points_np[i]
            label = labels_np[i]
            
            line = f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f} {label}"
            
            if class_names and label in class_names:
                line += f" {class_names[label]}"
            
            f.write(line + "\n")

def decompose_episode(episode_file, output_dir=None):
    """
    Zerlegt eine Episode in ihre Bestandteile und speichert sie als Punktwolken.
    
    Args:
        episode_file: Pfad zur .pt Episode-Datei
        output_dir: Ausgabeverzeichnis (optional)
    """
    # Lade Episode
    episode_data = load_episode(episode_file)
    
    # Bestimme Ausgabeverzeichnis
    if output_dir is None:
        episode_name = Path(episode_file).stem
        output_dir = f"stats/episodes/{episode_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Klassen-Namen Mapping (aus outdoor_fs.py)
    CLASS_ID_TO_NAME = {
        0: "Andere",
        1: "StreetSign", 
        2: "TrafficSign",
        3: "CircularTrafficSign",
        4: "OctagonalTrafficSign", 
        5: "RectangularTrafficSign",
        6: "TriangularTrafficSign",
        7: "DirectionSign",
        8: "FlippedTriangularTrafficSign",
        9: "PriorityRoad",
        10: "OneWayStreet",
        11: "Zone",
        12: "Exit",
        13: "DistanceMarker",
    }
    
    sampled_classes = episode_data['sampled_classes']
    print(f"Episode mit Klassen: {sampled_classes}")
    print(f"Speichere Bestandteile nach: {output_dir}")
    
    # Support-Sets verarbeiten
    support_feat = episode_data['support_feat']
    support_label = episode_data['support_label']
    
    if isinstance(support_feat, list):
        # Mehrere Support-Sets
        for i, (feat, label) in enumerate(zip(support_feat, support_label)):
            filename = os.path.join(output_dir, f"support_{i:02d}.txt")
            save_pointcloud_as_txt(feat, label, filename, CLASS_ID_TO_NAME)
            print(f"Support-Set {i}: {len(feat)} Punkte -> {filename}")
    else:
        # Einzelnes Support-Set
        filename = os.path.join(output_dir, "support.txt")
        save_pointcloud_as_txt(support_feat, support_label, filename, CLASS_ID_TO_NAME)
        print(f"Support-Set: {len(support_feat)} Punkte -> {filename}")
    
    # Query-Sets verarbeiten  
    query_feat = episode_data['query_feat']
    query_label = episode_data['query_label']
    
    if isinstance(query_feat, list):
        # Mehrere Query-Sets
        for i, (feat, label) in enumerate(zip(query_feat, query_label)):
            filename = os.path.join(output_dir, f"query_{i:02d}.txt")
            save_pointcloud_as_txt(feat, label, filename, CLASS_ID_TO_NAME)
            print(f"Query-Set {i}: {len(feat)} Punkte -> {filename}")
    else:
        # Einzelnes Query-Set
        filename = os.path.join(output_dir, "query.txt")
        save_pointcloud_as_txt(query_feat, query_label, filename, CLASS_ID_TO_NAME)
        print(f"Query-Set: {len(query_feat)} Punkte -> {filename}")
    
    # Metadaten speichern
    metadata_file = os.path.join(output_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Episode-Datei: {episode_file}\n")
        f.write(f"Sampled Classes: {sampled_classes}\n")
        f.write(f"Support Sets: {len(support_feat) if isinstance(support_feat, list) else 1}\n")
        f.write(f"Query Sets: {len(query_feat) if isinstance(query_feat, list) else 1}\n")
        
        if isinstance(support_feat, list):
            for i, feat in enumerate(support_feat):
                f.write(f"Support {i}: {len(feat)} Punkte\n")
        else:
            f.write(f"Support: {len(support_feat)} Punkte\n")
            
        if isinstance(query_feat, list):
            for i, feat in enumerate(query_feat):
                f.write(f"Query {i}: {len(feat)} Punkte\n")
        else:
            f.write(f"Query: {len(query_feat)} Punkte\n")
    
    print(f"Metadaten gespeichert: {metadata_file}")
    print(f"Episode erfolgreich zerlegt!")

def main():
    if len(sys.argv) < 2:
        print("Verwendung: python decompose_episode.py <episode_file.pt> [output_dir]")
        print("Beispiel: python decompose_episode.py test_episode.pt")
        print("Beispiel: python decompose_episode.py test_episode.pt custom_output/")
        sys.exit(1)
    
    episode_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        decompose_episode(episode_file, output_dir)
    except Exception as e:
        print(f"Fehler beim Zerlegen der Episode: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


