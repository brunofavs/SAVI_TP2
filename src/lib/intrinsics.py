#!/usr/bin/env python3
# Generate JSON with instrinsic values

import json
import os
import numpy as np


def main():
    
    # Camera Intrinsics
    fx = 570
    fy = 570
    cx = 320
    cy = 240
    width = 640
    height = 480
    intrinsic_matrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


    with open(f'{os.getenv("SAVI_TP2")}/src/lib/jsons/intrinsic.json','w') as f:
        json.dump(intrinsic_matrix.tolist(),f)

    print("Information saved!")

if __name__ == "__main__":
    main()