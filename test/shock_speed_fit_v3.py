import numpy as np, struct, glob, re
from scipy.optimize import brentq
from scipy.stats import linregress

def read_geo_coords(fname):
    with open(fname,'rb') as f:
        for _ in range(5): f.read(80)
        f.read(80); struct.unpack('<i', f.read(4)); f.read(80); f.read(80)
        nx, ny, nz = struct.unpack('<3i', f.read(12))
        n = nx*ny*nz
        X = np.frombuffer(f.read(n*4), dtype='<f4').reshape(nz,ny,nx)
        f.read(2*n*4)
        return nx, ny, nz, X

def read_rho_row(fname, nx, ny, j):
    with open(fname,'rb') as f:
        f.read(80); f.read(80); struct.unpack('<i', f.read(4)); f.read(80)
        return np.frombuffer(f.read(nx*ny*4), dtype='<f4').reshape(ny, nx)[j, :]

def get_time(case_file):
    with open(case_file) as f:
        for line in f:
            if 'Time' in line: return float(line.split()[-1])

def front_by_gradient_peak(x_row, rho_row):
    grad = np.abs(np.diff(rho_row)) / np.diff(x_row)
    k = np.argmax(grad)
    if k == 0 or k == len(grad)-1: return None
    y0, y1, y2 = grad[k-1], grad[k], grad[k+1]
    denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-12: return x_row[k]
    delta = 0.5*(y0 - y2)/denom
    xk = 0.5*(x_row[k]+x_row[k+1])
    dx = x_row[k+1]-x_row[k]
    return xk + delta*dx

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

nx, ny, nz, X = read_geo_coords('field.geo')
j_mid = ny//2
x_row = X[0, j_mid, :]
print(f"Grid: {nx}x{ny}   Theory S_R={S_R_theory:.5f}")

files = sorted(glob.glob('field?????.rho'))
times, fronts = [], []
for fn in files:
    n = re.search(r'(\d+)', fn).group(1)
    t = get_time(f'field{n}.case')
    if t is None or t > 0.15: continue
    rho_row = read_rho_row(fn, nx, ny, j_mid)
    xf = front_by_gradient_peak(x_row, rho_row)
    if xf is not None:
        times.append(t); fronts.append(xf)

times, fronts = np.array(times), np.array(fronts)
slope, intercept, r, p, se = linregress(times, fronts)
print(f"N={len(times)}  Fitted speed={slope:.5f}  R^2={r**2:.6f}  err={100*abs(slope-S_R_theory)/S_R_theory:.2f}%")