import pandas as pd

df = pd.read_csv("ibm_map_dump.csv")

# Group by the absolute value of Ny to find symmetric pairs
df['abs_Ny'] = df['Ny'].abs().round(5)
asymmetries = 0

for abs_ny, group in df.groupby('abs_Ny'):
    if abs_ny == 0.0: continue
    
    mirrors = group['is_mirror'].unique()
    cfs = group['cf'].unique()
    
    if len(mirrors) > 1 or len(cfs) > 1:
        print(f"ASYMMETRY DETECTED at |Ny| = {abs_ny}")
        print(group[['ghost_idx', 'is_mirror', 'cf', 'Ny']])
        asymmetries += 1

if asymmetries == 0:
    print("Map is PERFECTLY symmetric. The bug is strictly inside the GPU math kernel.")