import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ============================================================
# USER INPUT SECTION WITH REYNOLDS NUMBER GUIDANCE
# ============================================================
print("==========================================================")
print("              3D AIRFLOW AROUND A SPHERE                  ")
print("==========================================================")
print("Reynolds Number (Re) Flow Regimes:")
print("  • Re < 20    : Pure Steady Laminar Flow (No Wake)")
print("  • 20 - 199   : Steady Recirculating Ring Wake")
print("  • 200 - 999  : Unsteady Periodic Vortex Shedding")
print("  • Re >= 1000 : Fully Chaotic Turbulent Wake & Fluctuations")
print("----------------------------------------------------------")

try:
    user_re_input = input("Enter Reynolds Number (e.g. 10, 100, 500, 2500) [Default: 2500]: ")
    if user_re_input.strip() == "":
        Re = 2500.0
    else:
        Re = float(user_re_input)
except ValueError:
    print("Invalid input! Defaulting to Re = 2500.0")
    Re = 2500.0

print(f"\n[INFO] Simulating flow regime for Reynolds Number (Re) = {Re}...")

# Simulation Constants
U = 40.0
R = 8.0

xmin, xmax = -20, 50
ymin, ymax = -20, 20
zmin, zmax = -10, 20

# Streamline Grid Seed Setup
y_seeds = np.linspace(-18, 18, 14)
z_seeds = np.linspace(-8, 18, 10)
Y0, Z0 = np.meshgrid(y_seeds, z_seeds)
y_seeds_flat = Y0.flatten()
z_seeds_flat = Z0.flatten()

num_frames = 60
output_gif = f"airflow_sphere_Re_{int(Re)}.gif"


def generate_full_streamline(y0, z0, t, Re):
    """Integrates streamlines continuously across the entire X domain [-20, 50]."""
    x = xmin
    y = y0
    z = z0
    pts = []
    dx = 0.4  # Integration step size

    for step_idx in range(int((xmax - xmin) / dx) + 15):
        # Stop integration if boundaries are exceeded cleanly
        if x > xmax or abs(y) > ymax or abs(z) > zmax:
            break

        r = np.sqrt(x**2 + y**2 + z**2)

        # Deflect around sphere boundary smoothly
        if r < R + 0.15:
            factor = (R + 0.2) / max(r, 0.001)
            x *= factor
            y *= factor
            z *= factor
            r = R + 0.2

        pts.append([x, y, z])

        # 1. Base Potential Flow Field
        q = (R / r) ** 3
        vx = U * (1 + 0.5 * q - 1.5 * q * (x**2) / (r**2))
        vy = -1.5 * U * q * x * y / (r**2)
        vz = -1.5 * U * q * x * z / (r**2)

        # 2. REYNOLDS NUMBER DEPENDENT PHYSICAL REGIMES
        if Re < 20:
            # REGIME 1: Pure Steady Laminar Flow
            pass

        elif 20 <= Re < 200:
            # REGIME 2: Steady Recirculating Wake
            if x > R and (y**2 + z**2) < (1.4 * R)**2:
                wake_factor = np.exp(-((x - R) / 12)**2)
                vx -= 0.7 * U * wake_factor
                vy += 0.2 * U * (y / R) * wake_factor
                vz += 0.2 * U * (z / R) * wake_factor

        elif 200 <= Re < 1000:
            # REGIME 3: Unsteady Periodic Vortex Shedding
            if x > R:
                strouhal = 0.21
                amp = 0.35 * (Re / 500.0)
                decay = np.exp(-((x - R) / 25)**2) * np.exp(-(y**2 + z**2) / 110)
                phase = 2 * np.pi * strouhal * (U / (2 * R)) * t - 0.35 * (x - R)
                vy += amp * U * decay * np.sin(phase)
                vz += amp * U * decay * np.cos(phase)

        else:
            # REGIME 4: High Re / Turbulent Flow (Re >= 1000)
            if x > R:
                strouhal = 0.32
                amp = min(0.8, 0.4 + (Re - 1000) / 4000.0)
                decay = np.exp(-((x - R) / 30)**2) * np.exp(-(y**2 + z**2) / 120)

                phase1 = 2 * np.pi * strouhal * (U / (2 * R)) * t - 0.40 * (x - R)
                phase2 = 2 * np.pi * (strouhal * 2.1) * t - 0.80 * (x - R) + 0.5
                phase3 = 2 * np.pi * (strouhal * 3.7) * t - 1.20 * (x - R) + 1.2

                turb_y = np.sin(phase1) + 0.35 * np.sin(phase2) + 0.20 * np.cos(phase3)
                turb_z = np.cos(phase1) + 0.35 * np.cos(phase2) + 0.20 * np.sin(phase3)

                vy += amp * U * decay * turb_y
                vz += amp * U * decay * turb_z
                vx -= 0.25 * U * decay * np.abs(turb_y)

        # Ensure safe forward step calculations
        vx_safe = max(abs(vx), 1.0) * np.sign(vx if vx != 0 else 1)

        # Correct Forward Step Integration
        x += dx
        y += dx * (vy / vx_safe)
        z += dx * (vz / vx_safe)  # FIXED TYPO HERE!

    return np.array(pts)


# --- Figure Setup ---
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111, projection="3d")

# Sphere Mesh
theta = np.linspace(0, 2 * np.pi, 50)
phi = np.linspace(0, np.pi, 30)
theta, phi = np.meshgrid(theta, phi)
sx = R * np.sin(phi) * np.cos(theta)
sy = R * np.sin(phi) * np.sin(theta)
sz = R * np.cos(phi)


def update(frame):
    ax.clear()

    # 1. Render Sphere
    ax.plot_surface(
        sx, sy, sz, color="gray", linewidth=0, antialiased=True, alpha=0.9, shade=True
    )

    t = (frame / num_frames) * 3.0

    # 2. Compute Streamlines
    streams = []
    for y0, z0 in zip(y_seeds_flat, z_seeds_flat):
        s = generate_full_streamline(y0, z0, t, Re)
        if len(s) > 5:
            streams.append(s)

    # 3. Draw Cyan Streamlines
    for pts in streams:
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(seg, colors="#00d5ff", linewidth=0.9, alpha=0.85)
        ax.add_collection3d(lc)

    # 4. Formatting
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_zlabel("Z (m)", fontsize=11)

    if Re < 20:
        regime_str = "Pure Laminar"
    elif Re < 200:
        regime_str = "Recirculating Ring Wake"
    elif Re < 1000:
        regime_str = "Periodic Vortex Shedding"
    else:
        regime_str = "Chaotic Turbulent Wake"

    ax.set_title(
        f"3D Airflow Around Sphere (Re = {Re}) — [{regime_str}]",
        fontsize=14,
        pad=14,
    )

    ax.view_init(elev=20, azim=-70)
    ax.grid(True, alpha=0.25)


print(f"Generating GIF animation for Re = {Re}...")
ani = FuncAnimation(fig, update, frames=num_frames, interval=50)

writer = PillowWriter(fps=20)
ani.save(output_gif, writer=writer)
plt.close(fig)

print("==========================================================")
print(f"Done! Saved to: {output_gif}")
print("==========================================================")

##### OUTPUT
==========================================================
              3D AIRFLOW AROUND A SPHERE                  
==========================================================
Reynolds Number (Re) Flow Regimes:
  • Re < 20    : Pure Steady Laminar Flow (No Wake)
  • 20 - 199   : Steady Recirculating Ring Wake
  • 200 - 999  : Unsteady Periodic Vortex Shedding
  • Re >= 1000 : Fully Chaotic Turbulent Wake & Fluctuations
----------------------------------------------------------
Enter Reynolds Number (e.g. 10, 100, 500, 2500) [Default: 2500]:  800

[INFO] Simulating flow regime for Reynolds Number (Re) = 800.0...
Generating GIF animation for Re = 800.0...
==========================================================
Done! Saved to: airflow_sphere_Re_800.gif
==========================================================
