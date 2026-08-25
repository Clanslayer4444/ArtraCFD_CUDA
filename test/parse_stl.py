#!/usr/bin/env python3
import struct
import sys
import numpy as np

def read_binary_stl(path):
    with open(path, 'rb') as f:
        header = f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]
        print(f"Header: {header[:40]}...")
        print(f"Triangle count: {n_tri}")
        verts = []
        for _ in range(n_tri):
            normal = struct.unpack('<3f', f.read(12))
            v1 = struct.unpack('<3f', f.read(12))
            v2 = struct.unpack('<3f', f.read(12))
            v3 = struct.unpack('<3f', f.read(12))
            f.read(2)  # attribute byte count
            verts.extend([v1, v2, v3])
        return np.array(verts)

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'artracfd.stl'
    verts = read_binary_stl(path)
    unique_verts = np.unique(verts.round(6), axis=0)
    print(f"\nUnique vertices ({len(unique_verts)}):")
    for v in unique_verts:
        print(f"  {v}")
    print(f"\nBounding box: X[{verts[:,0].min():.4f}, {verts[:,0].max():.4f}]  "
          f"Y[{verts[:,1].min():.4f}, {verts[:,1].max():.4f}]  "
          f"Z[{verts[:,2].min():.4f}, {verts[:,2].max():.4f}]")