#!/usr/bin/env python3
"""
Quick reader for ArtraCFD's Ensight Gold BINARY structured scalar files.
Format (per ensight_writer.c):
  [80-byte str] "scalar variable"
  for each part:
    [80-byte str] "part"
    [4-byte int]  part number
    [80-byte str] dtype ("block" etc)
    [nx*ny*nz floats, IJK order, float32, native/little-endian]

Usage:
    python3 read_ensight.py field00050.p
    python3 read_ensight.py field00050.rho --dims 143 105 1   # if you know nx,ny,nz
    python3 read_ensight.py field00050.p --probe 137 115 135  # dump i=137, j=115..135, k=0
"""
import struct
import sys
import numpy as np
import argparse

ENSTR = 80

def read_string(f):
    raw = f.read(ENSTR)
    if len(raw) < ENSTR:
        return None
    return raw.split(b'\x00', 1)[0].decode('ascii', errors='replace')

def parse_scalar_file(path):
    parts = []
    with open(path, 'rb') as f:
        header = read_string(f)
        print(f"[header] {header!r}")
        while True:
            tag = read_string(f)
            if tag is None:
                break
            if tag.strip() != 'part':
                print(f"WARNING: expected 'part', got {tag!r} -- stopping")
                break
            part_num_bytes = f.read(4)
            if len(part_num_bytes) < 4:
                break
            part_num = struct.unpack('<i', part_num_bytes)[0]
            dtype = read_string(f)
            print(f"[part {part_num}] dtype={dtype!r}")
            # We don't know nx,ny,nz a priori from this file alone (Ensight relies on
            # the matching .geo file for that). We'll read floats until EOF or next
            # 'part' tag -- but since we can't peek cheaply, read all remaining floats
            # for THIS part assuming single-part file (most common case here).
            rest = f.read()
            # Try to detect if there's a trailing 'part' tag (multi-part) -- rare here.
            n_floats = len(rest) // 4
            data = np.frombuffer(rest[:n_floats*4], dtype='<f4')
            parts.append((part_num, dtype, data))
            print(f"  -> {n_floats} float values read")
            break  # single-part assumption; remove this break if you have multiple parts
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--dims', nargs=3, type=int, metavar=('NX','NY','NZ'),
                     help='grid dims to reshape into (IJK order, i fastest)')
    ap.add_argument('--probe', nargs=3, type=int, metavar=('I','J_MIN','J_MAX'),
                     help='print values at fixed i, j range j_min..j_max, k=0')
    ap.add_argument('--proberow', nargs=3, type=int, metavar=('J','I_MIN','I_MAX'),
                     help='print values at fixed j, i range i_min..i_max, k=0')
    ap.add_argument('--argmax', action='store_true',
                     help='report i,j,k location of the max value (requires --dims)')
    ap.add_argument('--argmin', action='store_true',
                     help='report i,j,k location of the min value (requires --dims)')
    args = ap.parse_args()

    parts = parse_scalar_file(args.file)
    if not parts:
        print("No data parsed.")
        return
    part_num, dtype, data = parts[0]

    print(f"\nTotal values: {data.size}")
    print(f"Min: {data.min():.6f}  Max: {data.max():.6f}  Mean: {data.mean():.6f}")

    if args.dims:
        nx, ny, nz = args.dims
        if nx*ny*nz != data.size:
            print(f"WARNING: dims {nx}x{ny}x{nz}={nx*ny*nz} != data.size={data.size}")
        else:
            arr = data.reshape((nz, ny, nx))  # k,j,i order since i fastest
            if args.argmax:
                k, j, i = np.unravel_index(np.argmax(arr), arr.shape)
                print(f"\nArgmax: i={i}, j={j}, k={k}  value={arr[k,j,i]:.6f}")
            if args.argmin:
                k, j, i = np.unravel_index(np.argmin(arr), arr.shape)
                print(f"\nArgmin: i={i}, j={j}, k={k}  value={arr[k,j,i]:.6f}")
            if args.probe:
                i, jmin, jmax = args.probe
                print(f"\nProbe at i={i}, j={jmin}..{jmax}, k=0:")
                for j in range(jmin, jmax+1):
                    if 0 <= j < ny and 0 <= i < nx:
                        print(f"  j={j:03d}  value={arr[0, j, i]:.6f}")
            if args.proberow:
                j, imin, imax = args.proberow
                print(f"\nProbe at j={j}, i={imin}..{imax}, k=0:")
                for i in range(imin, imax+1):
                    if 0 <= j < ny and 0 <= i < nx:
                        print(f"  i={i:03d}  value={arr[0, j, i]:.6f}")

if __name__ == '__main__':
    main()