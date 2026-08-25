import numpy as np, struct, glob, re
from scipy.optimize import brentq
from scipy.stats import linregress

# --- read Ensight Gold binary structured geometry: coords in block format (all X, then all Y, then all Z) ---
def read_geo_coords(fname):
    with open(fname,'rb') as f:
        f.read(80)                       # 'C Binary'
        f.read(80)                       # description line 1
        f.read(80)                       # description line 2
        f.read(80)                       # 'node id off'
        f.read(80)                       # 'element id off'
        f.read(80)                       # 'part'
        struct.unpack('<i', f.read(4))   # part number
        f.read(80)                       # part description
        f.read(80)                       # 'block'
        nx, ny, nz = struct.unpack('<3i', f.read(12))
        n = nx*ny*nz
        X = np.frombuffer(f.read(n*4), dtype='<f4')
        Y = np.frombuffer(f.read(n*4), dtype='<f4')
        Z = np.frombuffer(f.read(n*4), dtype='<f4')
        return nx, ny, nz, X.reshape(nz,ny,nx), Y.reshape(nz,ny,nx), Z.reshape(nz,ny,nx)

nx, ny, nz, X, Y, Z = read_geo_coords('field.geo')
print(f"Grid from field.geo: nx={nx} ny={ny} nz={nz}")
j_mid = ny//2
x_row = X[0, j_mid, :]   # exact physical x for each i at mid-row
print(f"x_row[0]={x_row[0]:.5f}  x_row[-1]={x_row[-1]:.5f}  dx(actual)={x_row[1]-x_row[0]:.6f}")

gamma = 1.4
rho_L, u_L, p_L = 3.67372, 2.41981, 9.04545
rho_R, u_R, p_R = 1.0, 0.0, 1.0
c_L = np.sqrt(gamma*p_L/rho_L); c_R = np.sqrt(gamma*p_R/rho_R)

def f_K(p, rho_K, p_K, c_K):
    A_K = 2.0/((gamma+1)*rho_K); B_K = (gamma-1)/(gamma+1)*p_K
    if p > p_K: return (p-p_K)*np.sqrt(A_K/(p+B_K))
    return (2*c_K/(gamma-1))*((p/p_K)**((gamma-1)/(2*gamma))-1)

p_star = brentq(lambda p: f_K(p,rho_L,p_L,c_L)+f_K(p,rho_R,p_R,c_R)+(u_R-u_L), 1e-6, 1000)
q_R = np.sqrt((gamma+1)/(2*gamma)*(p_star/p_R)+(gamma-1)/(2*gamma))
S_R_theory = u_R + c_R*q_R
print(f"\nTheoretical shock speed: {S_R_theory:.5f}")

def read_rho_row(fname, nx, ny, j):
    with open(fname,'rb') as f:
        f.read(80); f.read(80); struct.unpack('<i', f.read(4)); f.read(80)
        data = np.frombuffer(f.read(nx*ny*4), dtype='<f4').reshape(ny, nx)
    return data[j, :]

def get_time(case_file):
    with open(case_file) as f:
        for line in f:
            if 'Time' in line: return float(line.split()[-1])

threshold = 0.5*(rho_L+rho_R)
files = sorted(glob.glob('field?????.rho'))
times, fronts = [], []
for fn in files:
    n = re.search(r'(\d+)', fn).group(1)
    t = get_time(f'field{n}.case')
    if t is None or t > 0.15:
        continue
    row = read_rho_row(fn, nx, ny, j_mid)
    idx = None
    for k in range(len(row)-1):
        if row[k] >= threshold >= row[k+1]:
            frac = (row[k]-threshold)/(row[k]-row[k+1])
            idx = k, frac
            break
    if idx is not None:
        k, frac = idx
        # interpolate EXACT physical x using real coordinates, not assumed dx
        x_front = x_row[k] + frac*(x_row[k+1]-x_row[k])
        times.append(t); fronts.append(x_front)

times = np.array(times); fronts = np.array(fronts)
print(f"\nUsed {len(times)} snapshots, t in [{times.min():.4f}, {times.max():.4f}]")

slope, intercept, r, p, se = linregress(times, fronts)
print(f"Fitted shock speed (exact coords): {slope:.5f}")
print(f"R^2 = {r**2:.6f}   std err = {se:.5f}")
print(f"Relative error vs theory: {100*abs(slope-S_R_theory)/S_R_theory:.2f}%")