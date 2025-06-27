#!/bin/bash
python3 scripts/test_multipoint_loader.py \
    --config config/essen_outdoor_COSeg_fs.yaml \
    --epochs 3 \
    --episodes 2 \
    --save_dir datasets/test