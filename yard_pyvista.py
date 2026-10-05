"""
Container yard – 3D urgency view (PyVista)
pip install pyvista
"""
import numpy as np
import pyvista as pv

# ------------------------------------------------------------------
# Yard layout (real 20 ft ISO container proportions, metres)
# ------------------------------------------------------------------
BAYS, ROWS, TIERS = 7, 8, 4
CL, CW, CH = 6.06, 2.44, 2.59
GAP_X, GAP_Y = 0.6, 0.35            # spacing between bays / rows

# Urgency categories: (label, colour)
CATEGORIES = [
    ("Critical", "#ff1744"),
    ("Urgent",   "#ff7b00"),
    ("Moderate", "#ffd600"),
    ("Can wait", "#00e676"),
    ("Low",      "#00b0ff"),
]

def category(u):                    # u in 0..1, 1 = most urgent
    return min(int((1 - u) * len(CATEGORIES)), len(CATEGORIES) - 1)

# ------------------------------------------------------------------
# Sample data – replace with your own list of dicts
# ------------------------------------------------------------------
rng = np.random.default_rng(7)
containers = []
for bay in range(BAYS):
    for row in range(ROWS):
        for tier in range(rng.integers(0, TIERS + 1)):
            containers.append(dict(bay=bay, row=row, tier=tier,
                                   urgency=rng.random()))

# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------
def box_centre(c):
    return np.array([c["bay"] * (CL + GAP_X) + CL / 2,
                     c["row"] * (CW + GAP_Y) + CW / 2,
                     c["tier"] * CH + CH / 2])

def container_mesh(bay, row, tier):
    x0 = bay * (CL + GAP_X)
    y0 = row * (CW + GAP_Y)
    z0 = tier * CH
    return pv.Box(bounds=(x0, x0 + CL, y0, y0 + CW, z0, z0 + CH))

# one merged mesh per category = fast, and gives a clean legend
groups = {i: [] for i in range(len(CATEGORIES))}
for c in containers:
    groups[category(c["urgency"])].append(
        container_mesh(c["bay"], c["row"], c["tier"]))
meshes = {i: pv.merge(m) for i, m in groups.items() if m}

yard_w = BAYS * (CL + GAP_X)
yard_d = ROWS * (CW + GAP_Y)
ground = pv.Plane(center=(yard_w / 2, yard_d / 2, -0.01),
                  i_size=yard_w + 14, j_size=yard_d + 14)

# faint wireframe of the full block capacity (like your slide)
capacity = pv.Box(bounds=(-0.3, yard_w - GAP_X + 0.3,
                          -0.3, yard_d - GAP_Y + 0.3, 0, TIERS * CH))

OPACITY = 1.0          # 1 = solid, lower = see-through

def darker(hex_colour, f=0.6):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(int(r * f), int(g * f), int(b * f))

def build_scene(p, view_dir=(1, 1, 0.75)):
    p.add_mesh(ground, color="#b9bec4", ambient=0.5)
    p.add_mesh(capacity, style="wireframe", color="#8a9099",
               line_width=1, opacity=0.5)
    # draw far boxes first so translucent colours blend correctly
    vd = np.array(view_dir, float)
    order = sorted(containers, key=lambda c: np.dot(vd, box_centre(c)))
    for c in order:
        i = category(c["urgency"])
        label, colour = CATEGORIES[i]
        mesh = container_mesh(c["bay"], c["row"], c["tier"])
        # solid, bright faces
        p.add_mesh(mesh, color=colour, opacity=OPACITY,
                   smooth_shading=False, specular=0.4,
                   specular_power=30, ambient=0.4, diffuse=0.7)
        # crisp opaque outlines in a darker shade of the same colour
        p.add_mesh(mesh.extract_feature_edges(), color=darker(colour),
                   line_width=1.6)

# ------------------------------------------------------------------
# Views: four corner angles, so every side of the block is seen
# ------------------------------------------------------------------
cx, cy, cz = yard_w / 2, yard_d / 2, TIERS * CH / 2
dist = max(yard_w, yard_d) * 1.6
VIEWS = [
    ("North-east", ( 1,  1, 0.75)),
    ("North-west", (-1,  1, 0.75)),
    ("South-west", (-1, -1, 0.75)),
    ("South-east", ( 1, -1, 0.75)),
]

pv.global_theme.font.family = "arial"
p = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(2000, 1400),
               border=False)
for k, (name, (dx, dy, dz)) in enumerate(VIEWS):
    p.subplot(k // 2, k % 2)
    p.set_background("white")
    build_scene(p, (dx, dy, dz))
    n = np.array([dx, dy, dz]) / np.linalg.norm([dx, dy, dz])
    p.camera.position = (cx + n[0] * dist, cy + n[1] * dist, cz + n[2] * dist)
    p.camera.focal_point = (cx, cy, cz * 0.6)
    p.camera.up = (0, 0, 1)
    p.enable_parallel_projection()           # diagram-like, no distortion
    p.reset_camera(bounds=capacity.bounds)
    p.camera.zoom(1.25)
    p.add_text(f"View from {name}", position="upper_left",
               font_size=12, color="#333333")
    if k == 1:
        p.add_legend(labels=[[l, c] for l, c in CATEGORIES], bcolor="white", border=True, size=(0.22, 0.28),
                     loc="upper right", face="rectangle")

p.screenshot("container_yard_pyvista.png")
# p.show()   # uncomment (and remove off_screen=True) for interactive rotation
