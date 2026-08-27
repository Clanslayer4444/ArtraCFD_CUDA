import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.linewidth': 1.0, 'axes.grid': True,
    'grid.alpha': 0.3, 'legend.frameon': True,
    'legend.fontsize': 9, 'figure.dpi': 150
})

# ---------------------------------------------------------------------------
# Plot 1: GPU vs CPU crossover (final, with clean 799x799 point)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot([9801], [112.0], 'o', color='tab:blue', ms=8, label='GPU (measured)')
ax.plot([9801], [87.86], 's', color='tab:orange', ms=8, label='CPU (measured)')

ax.plot([40000, 40000], [313.86, 445.52], '-', color='tab:blue', lw=1.2, alpha=0.5)
ax.plot([40000], [np.mean([313.86, 445.52])], 'o', color='tab:blue', ms=8, alpha=0.6,
        markerfacecolor='none', markeredgewidth=1.5)
ax.plot([40000], [359.75], 's', color='tab:orange', ms=8)

ax.plot([250000], [273.04], 'o', color='tab:blue', ms=8, label='GPU (measured)' if False else None)
ax.plot([250000], [2763.32], 's', color='tab:orange', ms=8)

ax.plot([640000], [4021.03], '^', color='tab:blue', ms=9,
        markerfacecolor='none', markeredgewidth=1.5,
        label='GPU, thermal-throttled (excluded from trend)')
ax.plot([640000], [6137.54], 's', color='tab:orange', ms=8, alpha=0.4)

# Clean, cooled 799x799 pair -- filled markers, same shapes as main series, larger size for emphasis
ax.plot([638401], [3584.46], 'o', color='tab:blue', ms=12,
        markeredgecolor='k', markeredgewidth=0.8, label='GPU, cooled (799x799)')
ax.plot([638401], [5950.77], 's', color='tab:orange', ms=12,
        markeredgecolor='k', markeredgewidth=0.8, label='CPU (799x799)')

ax.annotate('1.66x speedup', xy=(638401, 4600), fontsize=9, ha='center',
            style='italic', color='dimgray')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Grid size (cells)')
ax.set_ylabel('Wall-clock runtime (s)')
ax.set_title('ArtraCFD: GPU vs. CPU runtime vs. grid size\n(NVIDIA RTX 3060 Laptop GPU)')
ax.legend(loc='upper left', fontsize=8)
fig.tight_layout()
fig.savefig('/home/claude/crossover_plot_final.png')
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: Grid convergence (shock-speed error vs grid size, log-log)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))

N = np.array([199, 399, 799])
err = np.array([1.05, 0.76, 0.17])

ax.loglog(N, err, 'o-', color='tab:blue', ms=9, lw=1.5, label='Measured error')

# fit observed order of convergence: err ~ N^-p
p, logC = np.polyfit(np.log(N), np.log(err), 1)
fit = np.exp(logC) * N.astype(float)**p
ax.loglog(N, fit, '--', color='gray', lw=1.2,
          label=f'Fit: observed order p = {-p:.2f}')

ax.set_xlabel('Grid resolution (N x N)')
ax.set_ylabel('Shock speed error vs. exact Riemann solution (%)')
ax.set_title('Grid convergence: measured shock speed\nvs. exact analytical solution')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig('/home/claude/convergence_plot.png')
plt.close(fig)

print(f"Observed convergence order p = {-p:.3f}")
print("Both plots saved.")