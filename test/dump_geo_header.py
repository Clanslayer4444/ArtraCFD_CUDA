import struct

with open('field.geo', 'rb') as f:
    data = f.read(400)  # first several header lines worth

# Print each 80-byte chunk as text, and try to spot where floats/ints begin
for i in range(6):
    chunk = data[i*80:(i+1)*80]
    text = chunk.split(b'\x00')[0].split(b'\x20\x20')[0]
    print(f"[{i*80:4d}] {chunk[:40]}")