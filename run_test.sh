#!/bin/bash
python3 scripts/test_multipoint_loader.py \
    --config config/essen_outdoor_COSeg_fs.yaml \
    --epochs 3 \
    --episodes 2 \
    --save_dir /sc/projects/sci-doellner/chair/adrian.schmidt/coseg_data/essen-road/epochs