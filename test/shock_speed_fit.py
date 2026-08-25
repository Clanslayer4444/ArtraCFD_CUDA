import numpy as np, struct, glob, re
from scipy.optimize import brentq
from scipy.stats import linregress

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
print(f"Theoretical shock speed: {S_R_theory:.5f}")

# --- read one .rho file, return 1D array along j=mid row ---
def read_rho_row(fname, nx, ny, j):
    with open(fname,'rb') as f:
        f.read(80)                      # 'scalar variable'
        f.read(80); struct.unpack('<i', f.read(4))  # 'part', part#
        f.read(80)                      # dtype
        data = np.frombuffer(f.read(nx*ny*4), dtype='<f4').reshape(ny, nx)
    return data[j, :]

def get_time(case_file):
    with open(case_file) as f:
        for line in f:
            if 'Time' in line: return float(line.split()[-1])

nx, ny = 98, 98
j_mid = ny//2
xmin, xmax, mx = -1.5, 3.0, 100
dx = (xmax-xmin)/mx
threshold = 0.5*(rho_L+rho_R)

files = sorted(glob.glob('field?????.rho'))
times, fronts = [], []
for fn in files:
    n = re.search(r'(\d+)', fn).group(1)
    t = get_time(f'field{n}.case')
    if t is None or t > 0.15:   # stop once shock likely reaches wedge region
        continue
    row = read_rho_row(fn, nx, ny, j_mid)
    # find first crossing below threshold scanning from left driver region
    idx = None
    for k in range(len(row)-1):
        if row[k] >= threshold >= row[k+1]:
            frac = (row[k]-threshold)/(row[k]-row[k+1])
            idx = k + frac
            break
    if idx is not None:
        times.append(t); fronts.append(idx*dx + xmin)

times = np.array(times); fronts = np.array(fronts)
print(f"\nUsed {len(times)} snapshots, t in [{times.min():.4f}, {times.max():.4f}]")

slope, intercept, r, p, se = linregress(times, fronts)
print(f"Fitted shock speed (linear regression): {slope:.5f}")
print(f"R^2 = {r**2:.6f}   std err = {se:.5f}")
print(f"Relative error vs theory: {100*abs(slope-S_R_theory)/S_R_theory:.2f}%")