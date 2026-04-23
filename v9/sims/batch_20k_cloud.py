"""
20,750-sim CalculiX sweep, Linux cloud variant.

  python3 batch_20k_cloud.py              # resumes from CSV
  python3 batch_20k_cloud.py --test 10    # smoke test
"""

import subprocess
import os
import sys
import math
import random
import csv
import time
import shutil
import argparse
import tempfile
from multiprocessing import Pool

PLATE_L = 100.0
PLATE_W = 50.0
NUM_SAMPLES = 20750
RANDOM_SEED = 101
MAX_DEFECTS = 5
MAX_PLACEMENT_ATTEMPTS = 200
MIN_CRACK_WIDTH = 0.15

CRACK_SEG_LEN_MIN = 0.2
CRACK_SEG_LEN_MAX = 0.8
MAX_ANGLE_DEV_DEG = 45.0
MIN_POLYGON_SEGMENTS = 12

LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]

E1, E2, E3 = 135000.0, 10000.0, 10000.0
NU12, NU13, NU23 = 0.27, 0.27, 0.45
G12, G13, G23 = 5200.0, 5200.0, 3900.0

XT, XC = 1500.0, 1200.0
YT, YC = 50.0, 250.0
SL, ST = 70.0, 35.0

CCX_EXE = "/usr/bin/ccx"
WORK_DIR = os.path.expanduser("~/sims")
OUTPUT_CSV = os.path.join(WORK_DIR, "calculix_results_20k.csv")
LOG_FILE = os.path.join(WORK_DIR, "batch_20k_cloud.log")

SOLVER_TIMEOUT = 300
CRACK_SEARCH_BUFFER = 3.0
NUM_WORKERS = 30
BACKUP_INTERVAL = 500
MAX_FAIL_RATE = 0.10
MIN_SIMS_FOR_FAIL_CHECK = 200

GLOBAL_RANGES = {
    'pressure_x':     [5.0,   100.0],
    'pressure_y':     [0.0,   100.0],
    'ply_thickness':  [0.10,  0.20],
    'layup_rotation': [0.0,   90.0],
}

DEFECT_RANGES = {
    'x':           [15.0, 85.0],
    'y':           [10.0, 40.0],
    'half_length': [4.0,  15.0],
    'width':       [0.15, 0.6],
    'angle':       [0.0,  180.0],
    'roughness':   [0.15, 0.90],
}

CSV_COLUMNS = [
    'sim_id', 'n_defects',
    'defect1_x', 'defect1_y', 'defect1_half_length', 'defect1_width', 'defect1_angle', 'defect1_roughness',
    'defect2_x', 'defect2_y', 'defect2_half_length', 'defect2_width', 'defect2_angle', 'defect2_roughness',
    'defect3_x', 'defect3_y', 'defect3_half_length', 'defect3_width', 'defect3_angle', 'defect3_roughness',
    'defect4_x', 'defect4_y', 'defect4_half_length', 'defect4_width', 'defect4_angle', 'defect4_roughness',
    'defect5_x', 'defect5_y', 'defect5_half_length', 'defect5_width', 'defect5_angle', 'defect5_roughness',
    'defect1_cos_angle', 'defect1_sin_angle', 'defect1_aspect_ratio', 'defect1_norm_x', 'defect1_norm_y',
    'defect1_norm_length', 'defect1_boundary_prox', 'defect1_ligament_ratio', 'defect1_sif_estimate',
    'defect2_cos_angle', 'defect2_sin_angle', 'defect2_aspect_ratio', 'defect2_norm_x', 'defect2_norm_y',
    'defect2_norm_length', 'defect2_boundary_prox', 'defect2_ligament_ratio', 'defect2_sif_estimate',
    'defect3_cos_angle', 'defect3_sin_angle', 'defect3_aspect_ratio', 'defect3_norm_x', 'defect3_norm_y',
    'defect3_norm_length', 'defect3_boundary_prox', 'defect3_ligament_ratio', 'defect3_sif_estimate',
    'defect4_cos_angle', 'defect4_sin_angle', 'defect4_aspect_ratio', 'defect4_norm_x', 'defect4_norm_y',
    'defect4_norm_length', 'defect4_boundary_prox', 'defect4_ligament_ratio', 'defect4_sif_estimate',
    'defect5_cos_angle', 'defect5_sin_angle', 'defect5_aspect_ratio', 'defect5_norm_x', 'defect5_norm_y',
    'defect5_norm_length', 'defect5_boundary_prox', 'defect5_ligament_ratio', 'defect5_sif_estimate',
    'pressure_x', 'pressure_y', 'ply_thickness', 'layup_rotation',
    'min_inter_defect_dist',
    'total_crack_area_frac', 'max_sif_estimate', 'min_ligament_ratio',
    'solver_completed', 'n_elements',
    'max_mises', 'max_s11', 'min_s11', 'max_s12',
    'tsai_wu_index',
    'max_hashin_ft', 'max_hashin_fc', 'max_hashin_mt', 'max_hashin_mc',
    'max_mises_defect1', 'max_mises_defect2', 'max_mises_defect3',
    'max_mises_defect4', 'max_mises_defect5',
    'failed_tsai_wu', 'failed_hashin',
]


_log_fh = None

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def latin_hypercube_sample(param_ranges, n_samples, seed=42):
    random.seed(seed)
    param_names = list(param_ranges.keys())
    columns = {}
    for name in param_names:
        lo, hi = param_ranges[name]
        samples = []
        for i in range(n_samples):
            stratum_lo = lo + (hi - lo) * i / n_samples
            stratum_hi = lo + (hi - lo) * (i + 1) / n_samples
            samples.append(random.uniform(stratum_lo, stratum_hi))
        random.shuffle(samples)
        columns[name] = samples
    samples_list = []
    for i in range(n_samples):
        sample = {}
        for name in param_names:
            sample[name] = columns[name][i]
        samples_list.append(sample)
    return samples_list


def validate_crack_bounds(cx, cy, half_length, width, angle_deg,
                          roughness, plate_length, plate_width, margin=2.0):
    max_lateral = width / 2.0 + half_length * 0.3
    t = math.radians(angle_deg)
    dx = abs(half_length * math.cos(t)) + abs(max_lateral * math.sin(t))
    dy = abs(half_length * math.sin(t)) + abs(max_lateral * math.cos(t))
    if cx - dx < margin or cx + dx > plate_length - margin:
        return False
    if cy - dy < margin or cy + dy > plate_width - margin:
        return False
    return True


def overlaps_existing(new_defect, existing_defects, margin=2.0):
    for d in existing_defects:
        dist = math.sqrt(
            (new_defect['x'] - d['x'])**2 +
            (new_defect['y'] - d['y'])**2)
        min_dist = new_defect['half_length'] + d['half_length'] + margin
        if dist < min_dist:
            return True
    return False


def place_defects_sequentially(n_defects):
    placed = []
    for crack_idx in range(n_defects):
        placed_this_crack = False
        for attempt in range(MAX_PLACEMENT_ATTEMPTS):
            defect = {}
            for pname, (lo, hi) in DEFECT_RANGES.items():
                defect[pname] = random.uniform(lo, hi)
            if defect['width'] < MIN_CRACK_WIDTH:
                defect['width'] = MIN_CRACK_WIDTH
            if not validate_crack_bounds(
                    defect['x'], defect['y'],
                    defect['half_length'], defect['width'],
                    defect['angle'], defect['roughness'],
                    PLATE_L, PLATE_W):
                continue
            if overlaps_existing(defect, placed):
                continue
            placed.append(defect)
            placed_this_crack = True
            break
        if not placed_this_crack:
            return None
    return placed


def generate_all_samples(n_total, seed=55):
    random.seed(seed)
    counts_per_n = n_total // MAX_DEFECTS
    remainder = n_total - counts_per_n * MAX_DEFECTS
    global_samples = latin_hypercube_sample(GLOBAL_RANGES, n_total, seed=seed)

    all_samples = []
    idx = 0
    placement_failures = 0
    MAX_SAMPLE_FAILURES = 50

    for n_def in range(1, MAX_DEFECTS + 1):
        n_sims = counts_per_n + (1 if n_def <= remainder else 0)
        placed_count = 0
        consecutive_failures = 0

        while placed_count < n_sims and idx < n_total:
            defects = place_defects_sequentially(n_def)
            if defects is None:
                placement_failures += 1
                consecutive_failures += 1
                if consecutive_failures >= MAX_SAMPLE_FAILURES:
                    break
                continue
            consecutive_failures = 0
            sample = {
                'n_defects': n_def,
                'pressure_x': global_samples[idx]['pressure_x'],
                'pressure_y': global_samples[idx]['pressure_y'],
                'ply_thickness': global_samples[idx]['ply_thickness'],
                'layup_rotation': global_samples[idx]['layup_rotation'],
                'defects': defects,
            }
            all_samples.append(sample)
            placed_count += 1
            idx += 1

    random.seed(seed + 999)
    random.shuffle(all_samples)
    return all_samples


def crack_polygon_points(cx, cy, half_length, width, angle_deg, roughness):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    max_dev_rad = math.radians(MAX_ANGLE_DEV_DEG) * roughness

    centerline = [(-half_length, 0.0)]
    x_pos = -half_length
    y_pos = 0.0

    while x_pos < half_length:
        seg_len = random.uniform(CRACK_SEG_LEN_MIN, CRACK_SEG_LEN_MAX)
        deviation = random.uniform(-max_dev_rad, max_dev_rad)
        dx = seg_len * math.cos(deviation)
        dy = seg_len * math.sin(deviation)
        x_pos += dx
        y_pos += dy
        y_pos = max(-half_length * 0.3, min(half_length * 0.3, y_pos))
        centerline.append((x_pos, y_pos))

    if len(centerline) > 2 and centerline[-1][0] > half_length * 1.05:
        px_prev, py_prev = centerline[-2]
        px_last, py_last = centerline[-1]
        dx_seg = px_last - px_prev
        if abs(dx_seg) > 1e-9:
            frac = (half_length - px_prev) / dx_seg
            frac = max(0.0, min(1.0, frac))
            new_y = py_prev + frac * (py_last - py_prev)
            centerline[-1] = (half_length, new_y)

    if len(centerline) < MIN_POLYGON_SEGMENTS // 2:
        refined = [centerline[0]]
        for j in range(1, len(centerline)):
            mid_x = (centerline[j - 1][0] + centerline[j][0]) / 2.0
            mid_y = (centerline[j - 1][1] + centerline[j][1]) / 2.0
            refined.append((mid_x, mid_y))
            refined.append(centerline[j])
        centerline = refined

    half_w = width / 2.0
    upper = []
    lower = []
    total_x_span = 2.0 * half_length
    if total_x_span < 1e-9:
        total_x_span = 1.0

    for i, (px, py) in enumerate(centerline):
        progress = (px + half_length) / total_x_span
        progress = max(0.0, min(1.0, progress))
        taper = 1.0 - (2.0 * abs(progress - 0.5)) ** 1.5
        taper = max(taper, 0.10)
        local_hw = half_w * taper
        upper.append((px, py + local_hw))
        lower.append((px, py - local_hw))

    local_polygon = upper + list(reversed(lower))

    global_points = []
    for lx, ly in local_polygon:
        gx = cx + lx * cos_a - ly * sin_a
        gy = cy + lx * sin_a + ly * cos_a
        global_points.append((round(gx, 6), round(gy, 6)))

    return global_points


def polygon_self_intersects(polygon):
    n = len(polygon)
    if n < 4:
        return False

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def segments_intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    for i in range(n):
        A = polygon[i]
        B = polygon[(i + 1) % n]
        for j in range(i + 2, n):
            if j == (i - 1) % n or (i == 0 and j == n - 1):
                continue
            C = polygon[j]
            D = polygon[(j + 1) % n]
            if segments_intersect(A, B, C, D):
                return True
    return False


def create_plate_with_cracks(polygons, job_name):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(job_name)

    plate = gmsh.model.occ.addRectangle(0, 0, 0, PLATE_L, PLATE_W)

    slot_surfs = []
    for di, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        pts = []
        for gx, gy in polygon:
            gx = max(0.01, min(PLATE_L - 0.01, gx))
            gy = max(0.01, min(PLATE_W - 0.01, gy))
            pts.append(gmsh.model.occ.addPoint(gx, gy, 0))
        lines = []
        n_pts = len(pts)
        for i in range(n_pts):
            lines.append(gmsh.model.occ.addLine(pts[i], pts[(i + 1) % n_pts]))
        try:
            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])
            slot_surfs.append((2, surf))
        except Exception:
            continue

    if slot_surfs:
        gmsh.model.occ.cut([(2, plate)], slot_surfs)

    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=5, name="plate")

    all_curves = gmsh.model.getEntities(1)
    tol = 0.1
    left_c, bottom_c, right_c, top_c = [], [], [], []
    for _, tag in all_curves:
        bbox = gmsh.model.getBoundingBox(1, tag)
        xmin, ymin, _, xmax, ymax, _ = bbox
        if abs(xmin) < tol and abs(xmax) < tol:
            left_c.append(tag)
        elif abs(xmin - PLATE_L) < tol and abs(xmax - PLATE_L) < tol:
            right_c.append(tag)
        elif abs(ymin) < tol and abs(ymax) < tol and xmax - xmin > 1.0:
            bottom_c.append(tag)
        elif abs(ymin - PLATE_W) < tol and abs(ymax - PLATE_W) < tol and xmax - xmin > 1.0:
            top_c.append(tag)

    if left_c:
        gmsh.model.addPhysicalGroup(1, left_c, tag=1, name="left")
    if bottom_c:
        gmsh.model.addPhysicalGroup(1, bottom_c, tag=2, name="bottom")
    if right_c:
        gmsh.model.addPhysicalGroup(1, right_c, tag=3, name="right")
    if top_c:
        gmsh.model.addPhysicalGroup(1, top_c, tag=4, name="top")

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 3.0)

    crack_curves = []
    for _, tag in all_curves:
        bbox = gmsh.model.getBoundingBox(1, tag)
        xmin, ymin, _, xmax, ymax, _ = bbox
        is_edge = (abs(xmin) < tol and abs(xmax) < tol) or \
                  (abs(xmin - PLATE_L) < tol and abs(xmax - PLATE_L) < tol) or \
                  (abs(ymin) < tol and abs(ymax) < tol and xmax - xmin > 5) or \
                  (abs(ymin - PLATE_W) < tol and abs(ymax - PLATE_W) < tol and xmax - xmin > 5)
        if not is_edge:
            crack_curves.append(tag)

    if crack_curves:
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "CurvesList", crack_curves)
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", 0.5)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", 3.0)
        gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.5)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", 15.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(ft)

    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = {}
    for i, tag in enumerate(node_tags):
        nodes[int(tag)] = (node_coords[3 * i], node_coords[3 * i + 1], node_coords[3 * i + 2])

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    elements = []
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        npe = gmsh.model.mesh.getElementProperties(etype)[3]
        for i, etag in enumerate(etags):
            enlist = [int(enodes[i * npe + j]) for j in range(npe)]
            elements.append((int(etag), npe, enlist))

    bc_sets = {}
    for phys_tag, name in [(1, "left"), (2, "bottom"), (3, "right"), (4, "top")]:
        nset = set()
        try:
            ents = gmsh.model.getEntitiesForPhysicalGroup(1, phys_tag)
            for ent in ents:
                ntags, _, _ = gmsh.model.mesh.getNodes(1, ent, includeBoundary=True)
                nset.update(int(t) for t in ntags)
        except:
            pass
        bc_sets[name] = nset

    corner = min(nodes.keys(), key=lambda n: nodes[n][0]**2 + nodes[n][1]**2)
    bc_sets["corner"] = {corner}

    gmsh.finalize()
    return nodes, elements, bc_sets


def write_ccx_inp(nodes, elements, bc_sets, case, job_name, work_dir=None):
    ply_t = case["ply_thickness"]
    total_t = 8 * ply_t
    if work_dir is None:
        work_dir = WORK_DIR
    filepath = os.path.join(work_dir, f"{job_name}.inp")

    with open(filepath, 'w') as f:
        f.write("** Auto-generated CalculiX input\n*HEADING\n")
        f.write(f"Batch sim {case['sim_id']}\n")

        f.write("*NODE\n")
        for nid in sorted(nodes.keys()):
            x, y, z = nodes[nid]
            f.write(f"  {nid}, {x:.8f}, {y:.8f}, {z:.8f}\n")

        f.write("*ELEMENT, TYPE=S6, ELSET=PLATE\n")
        for eid, npe, enlist in elements:
            node_str = ", ".join(str(n) for n in enlist)
            f.write(f"  {eid}, {node_str}\n")

        f.write("*MATERIAL, NAME=CFRP_UD\n")
        f.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
        f.write(f"{E1}, {E2}, {E3}, {NU12}, {NU13}, {NU23}, {G12}, {G13}\n")
        f.write(f"{G23}\n")

        f.write("*ORIENTATION, NAME=ORI_0, SYSTEM=RECTANGULAR\n")
        f.write("1.0, 0.0, 0.0, 0.0, 1.0, 0.0\n")
        f.write("*ORIENTATION, NAME=ORI_45, SYSTEM=RECTANGULAR\n")
        f.write("0.7071068, 0.7071068, 0.0, -0.7071068, 0.7071068, 0.0\n")
        f.write("*ORIENTATION, NAME=ORI_M45, SYSTEM=RECTANGULAR\n")
        f.write("0.7071068, -0.7071068, 0.0, 0.7071068, 0.7071068, 0.0\n")
        f.write("*ORIENTATION, NAME=ORI_90, SYSTEM=RECTANGULAR\n")
        f.write("0.0, 1.0, 0.0, -1.0, 0.0, 0.0\n")

        ori_names = {0: "ORI_0", 45: "ORI_45", -45: "ORI_M45", 90: "ORI_90"}
        f.write("*SHELL SECTION, COMPOSITE, ELSET=PLATE, OFFSET=0\n")
        for angle in LAYUP:
            f.write(f"{ply_t}, 3, CFRP_UD, {ori_names[angle]}\n")

        for name, nset in bc_sets.items():
            if nset:
                f.write(f"*NSET, NSET={name.upper()}\n")
                nids = sorted(nset)
                for k in range(0, len(nids), 16):
                    chunk = nids[k:k + 16]
                    f.write(", ".join(str(n) for n in chunk) + "\n")

        f.write("*BOUNDARY\nLEFT, 1, 1, 0.0\n")
        f.write("CORNER, 2, 3, 0.0\n")

        px = case['pressure_x']
        py = case['pressure_y']

        f.write("*STEP\n*STATIC\n")

        if bc_sets.get("right"):
            n_right = len(bc_sets["right"])
            force_per_node_x = (px * PLATE_W * total_t) / n_right
            f.write("*CLOAD\nRIGHT, 1, {:.8f}\n".format(force_per_node_x))

        if bc_sets.get("top"):
            n_top = len(bc_sets["top"])
            force_per_node_y_top = (py * PLATE_L * total_t) / n_top
            f.write("*CLOAD\nTOP, 2, {:.8f}\n".format(force_per_node_y_top))

        if bc_sets.get("bottom"):
            n_bottom = len(bc_sets["bottom"])
            force_per_node_y_bot = -(py * PLATE_L * total_t) / n_bottom
            f.write("*CLOAD\nBOTTOM, 2, {:.8f}\n".format(force_per_node_y_bot))

        f.write("*EL PRINT, ELSET=PLATE\nS\n")
        f.write("*END STEP\n")


def parse_stresses(dat_path):
    stress_data = []
    in_block = False
    with open(dat_path) as f:
        for line in f:
            if 'stresses (elem' in line:
                in_block = True
                continue
            if in_block:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        eid = int(parts[0])
                        ip = int(parts[1])
                        s11 = float(parts[2])
                        s22 = float(parts[3])
                        s33 = float(parts[4])
                        s12 = float(parts[5])
                        s13 = float(parts[6])
                        s23 = float(parts[7])
                        stress_data.append((eid, ip, s11, s22, s33, s12, s13, s23))
                    except (ValueError, IndexError):
                        if any(kw in line.lower() for kw in ['displacements', 'forces', 'step']):
                            in_block = False
    return stress_data


def percentile(sorted_list, p):
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    d = k - f
    return sorted_list[f] + d * (sorted_list[c] - sorted_list[f])


def compute_metrics(stress_data, element_centroids, defects):
    if not stress_data:
        return None

    F1 = 1.0 / XT - 1.0 / XC
    F2 = 1.0 / YT - 1.0 / YC
    F11 = 1.0 / (XT * XC)
    F22 = 1.0 / (YT * YC)
    F66 = 1.0 / (SL * SL)
    F12 = -0.5 * math.sqrt(F11 * F22)

    all_mises = []
    all_s11 = []
    all_s12 = []
    all_tw = []
    all_hft = []
    all_hfc = []
    all_hmt = []
    all_hmc = []
    elem_mises = {}

    for eid, ip, s11, s22, s33, s12, s13, s23 in stress_data:
        vm = math.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
                               + 6 * (s12**2 + s13**2 + s23**2)))
        all_mises.append(vm)
        all_s11.append(s11)
        all_s12.append(abs(s12))

        tw = F1 * s11 + F2 * s22 + F11 * s11**2 + F22 * s22**2 + F66 * s12**2 + 2 * F12 * s11 * s22
        all_tw.append(tw)

        hft = (s11 / XT)**2 + (s12 / SL)**2 if s11 > 0 else 0.0
        hfc = (s11 / XC)**2 if s11 < 0 else 0.0
        hmt = (s22 / YT)**2 + (s12 / SL)**2 if s22 > 0 else 0.0
        hmc = (s22 / (2 * ST))**2 + ((YC / (2 * ST))**2 - 1) * (s22 / YC) + (s12 / SL)**2 if s22 < 0 else 0.0

        all_hft.append(hft)
        all_hfc.append(hfc)
        all_hmt.append(hmt)
        all_hmc.append(hmc)

        if eid not in elem_mises:
            elem_mises[eid] = []
        elem_mises[eid].append(vm)

    all_mises_sorted = sorted(all_mises)
    all_s11_sorted = sorted(all_s11)
    all_s12_sorted = sorted(all_s12)
    all_tw_sorted = sorted(all_tw)
    all_hft_sorted = sorted(all_hft)
    all_hfc_sorted = sorted(all_hfc)
    all_hmt_sorted = sorted(all_hmt)
    all_hmc_sorted = sorted(all_hmc)

    max_mises = percentile(all_mises_sorted, 0.999)
    max_s11 = percentile(all_s11_sorted, 0.999)
    min_s11 = all_s11_sorted[0]
    max_s12 = percentile(all_s12_sorted, 0.999)
    tsai_wu_index = percentile(all_tw_sorted, 0.997)
    max_hashin_ft = percentile(all_hft_sorted, 0.998)
    max_hashin_fc = percentile(all_hfc_sorted, 0.998)
    max_hashin_mt = percentile(all_hmt_sorted, 0.998)
    max_hashin_mc = percentile(all_hmc_sorted, 0.998)

    failed_tsai_wu = 1 if tsai_wu_index >= 1.0 else 0
    failed_hashin = 1 if (max_hashin_ft >= 1.0 or max_hashin_fc >= 1.0 or
                          max_hashin_mt >= 1.0 or max_hashin_mc >= 1.0) else 0

    mises_per_defect = []
    for di in range(MAX_DEFECTS):
        if di < len(defects):
            d = defects[di]
            dcx, dcy = d['x'], d['y']
            local_max = 0.0
            for eid, mises_list in elem_mises.items():
                if eid in element_centroids:
                    ex, ey = element_centroids[eid]
                    dist = math.sqrt((ex - dcx)**2 + (ey - dcy)**2)
                    if dist <= CRACK_SEARCH_BUFFER + d['half_length']:
                        for vm in mises_list:
                            if vm > local_max:
                                local_max = vm
            mises_per_defect.append(local_max)
        else:
            mises_per_defect.append(0.0)

    return {
        'n_elements': len(set(eid for eid, *_ in stress_data)),
        'max_mises': max_mises,
        'max_s11': max_s11,
        'min_s11': min_s11,
        'max_s12': max_s12,
        'tsai_wu_index': tsai_wu_index,
        'max_hashin_ft': max_hashin_ft,
        'max_hashin_fc': max_hashin_fc,
        'max_hashin_mt': max_hashin_mt,
        'max_hashin_mc': max_hashin_mc,
        'mises_per_defect': mises_per_defect,
        'failed_tsai_wu': failed_tsai_wu,
        'failed_hashin': failed_hashin,
    }


def compute_element_centroids(nodes, elements):
    centroids = {}
    for eid, npe, enlist in elements:
        xs, ys = [], []
        for nid in enlist:
            if nid in nodes:
                xs.append(nodes[nid][0])
                ys.append(nodes[nid][1])
        if xs:
            centroids[eid] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return centroids


def compute_min_inter_defect_dist(defects):
    if len(defects) < 2:
        return 0.0
    min_dist = float('inf')
    for i in range(len(defects)):
        for j in range(i + 1, len(defects)):
            d = math.sqrt((defects[i]['x'] - defects[j]['x'])**2 +
                          (defects[i]['y'] - defects[j]['y'])**2)
            if d < min_dist:
                min_dist = d
    return min_dist


def compute_defect_features(d, pressure_x, pressure_y):
    x, y = d['x'], d['y']
    hl = d['half_length']
    w = d['width']
    angle_rad = math.radians(d['angle'])

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    aspect_ratio = hl / w if w > 1e-9 else 0.0
    norm_x = x / PLATE_L
    norm_y = y / PLATE_W
    norm_length = (2 * hl) / PLATE_W
    boundary_prox = min(x, PLATE_L - x, y, PLATE_W - y)

    crack_proj = 2 * hl * abs(sin_a)
    ligament_ratio = max(0.0, (PLATE_W - crack_proj) / PLATE_W)

    sigma = abs(pressure_x * sin_a**2 + pressure_y * cos_a**2)
    a = hl
    a_over_W = (2 * a) / PLATE_W
    if a_over_W < 0.95:
        F_corr = (1.0 - 0.025 * a_over_W**2 + 0.06 * a_over_W**4) / math.sqrt(math.cos(math.pi * a_over_W / 2))
    else:
        F_corr = 5.0
    sif = sigma * math.sqrt(math.pi * a) * F_corr

    return {
        'cos_angle': round(cos_a, 6),
        'sin_angle': round(sin_a, 6),
        'aspect_ratio': round(aspect_ratio, 6),
        'norm_x': round(norm_x, 6),
        'norm_y': round(norm_y, 6),
        'norm_length': round(norm_length, 6),
        'boundary_prox': round(boundary_prox, 6),
        'ligament_ratio': round(ligament_ratio, 6),
        'sif_estimate': round(sif, 6),
    }


def load_completed_sims(csv_path):
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    completed.add(int(row['sim_id']))
                except (ValueError, KeyError):
                    pass
    except Exception:
        pass
    return completed


def write_csv_header(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def append_csv_row(csv_path, row_dict):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        row = []
        for col in CSV_COLUMNS:
            row.append(row_dict.get(col, 0))
        writer.writerow(row)


def build_row(sim_id, sample, metrics=None, error=False, n_elements=0):
    row = {'sim_id': sim_id, 'n_defects': sample['n_defects']}

    all_sifs = []
    all_ligaments = []
    total_crack_area = 0.0

    for di in range(MAX_DEFECTS):
        prefix = f"defect{di+1}_"
        if di < len(sample['defects']):
            d = sample['defects'][di]
            row[prefix + 'x'] = round(d['x'], 6)
            row[prefix + 'y'] = round(d['y'], 6)
            row[prefix + 'half_length'] = round(d['half_length'], 6)
            row[prefix + 'width'] = round(d['width'], 6)
            row[prefix + 'angle'] = round(d['angle'], 6)
            row[prefix + 'roughness'] = round(d['roughness'], 6)

            feats = compute_defect_features(d, sample['pressure_x'], sample['pressure_y'])
            for fname, fval in feats.items():
                row[prefix + fname] = fval

            all_sifs.append(feats['sif_estimate'])
            all_ligaments.append(feats['ligament_ratio'])
            total_crack_area += d['half_length'] * d['width'] * 2
        else:
            for field in ['x', 'y', 'half_length', 'width', 'angle', 'roughness',
                          'cos_angle', 'sin_angle', 'aspect_ratio', 'norm_x', 'norm_y',
                          'norm_length', 'boundary_prox', 'ligament_ratio', 'sif_estimate']:
                row[prefix + field] = 0

    row['pressure_x'] = round(sample['pressure_x'], 6)
    row['pressure_y'] = round(sample['pressure_y'], 6)
    row['ply_thickness'] = round(sample['ply_thickness'], 6)
    row['layup_rotation'] = round(sample['layup_rotation'], 6)
    row['min_inter_defect_dist'] = round(compute_min_inter_defect_dist(sample['defects']), 6)

    plate_area = PLATE_L * PLATE_W
    row['total_crack_area_frac'] = round(total_crack_area / plate_area, 6)
    row['max_sif_estimate'] = round(max(all_sifs) if all_sifs else 0.0, 6)
    row['min_ligament_ratio'] = round(min(all_ligaments) if all_ligaments else 1.0, 6)

    if error or metrics is None:
        row['solver_completed'] = 'ERROR'
        row['n_elements'] = n_elements
        for col in ['max_mises', 'max_s11', 'min_s11', 'max_s12',
                     'tsai_wu_index', 'max_hashin_ft', 'max_hashin_fc',
                     'max_hashin_mt', 'max_hashin_mc']:
            row[col] = 0
        for di in range(MAX_DEFECTS):
            row[f'max_mises_defect{di+1}'] = 0
        row['failed_tsai_wu'] = 0
        row['failed_hashin'] = 0
    else:
        row['solver_completed'] = 'YES'
        row['n_elements'] = metrics['n_elements']
        row['max_mises'] = round(metrics['max_mises'], 6)
        row['max_s11'] = round(metrics['max_s11'], 6)
        row['min_s11'] = round(metrics['min_s11'], 6)
        row['max_s12'] = round(metrics['max_s12'], 6)
        row['tsai_wu_index'] = round(metrics['tsai_wu_index'], 6)
        row['max_hashin_ft'] = round(metrics['max_hashin_ft'], 6)
        row['max_hashin_fc'] = round(metrics['max_hashin_fc'], 6)
        row['max_hashin_mt'] = round(metrics['max_hashin_mt'], 6)
        row['max_hashin_mc'] = round(metrics['max_hashin_mc'], 6)
        for di in range(MAX_DEFECTS):
            row[f'max_mises_defect{di+1}'] = round(metrics['mises_per_defect'][di], 6)
        row['failed_tsai_wu'] = metrics['failed_tsai_wu']
        row['failed_hashin'] = metrics['failed_hashin']

    return row


def cleanup_large_files(job_name):
    for ext in ['.dat', '.frd', '.inp', '.12d', '.cvg', '.sta']:
        fpath = os.path.join(WORK_DIR, f"{job_name}{ext}")
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except OSError:
                pass


def run_single_sim(args):
    sim_id, sample, polygons = args
    job_name = f"b20k_sim{sim_id}"
    t0 = time.time()

    if polygons is None:
        return build_row(sim_id, sample, error=True)

    # isolated temp dir — CCX clobbers same-named files across workers
    tmp_dir = tempfile.mkdtemp(prefix=f"ccx_{sim_id}_", dir=WORK_DIR)

    try:
        nodes, elements, bc_sets = create_plate_with_cracks(polygons, job_name)
        n_elements = len(elements)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return build_row(sim_id, sample, error=True)

    case = {
        'sim_id': sim_id,
        'pressure_x': sample['pressure_x'],
        'pressure_y': sample['pressure_y'],
        'ply_thickness': sample['ply_thickness'],
    }
    write_ccx_inp(nodes, elements, bc_sets, case, job_name, work_dir=tmp_dir)

    try:
        subprocess.run(
            [CCX_EXE, job_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=SOLVER_TIMEOUT, cwd=tmp_dir)
        dat_check = os.path.join(tmp_dir, f"{job_name}.dat")
        if not os.path.exists(dat_check) or os.path.getsize(dat_check) < 100:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return build_row(sim_id, sample, error=True, n_elements=n_elements)
    except (subprocess.TimeoutExpired, Exception):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return build_row(sim_id, sample, error=True, n_elements=n_elements)

    dat_path = os.path.join(tmp_dir, f"{job_name}.dat")
    try:
        stress_data = parse_stresses(dat_path)
        if not stress_data:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return build_row(sim_id, sample, error=True, n_elements=n_elements)

        element_centroids = compute_element_centroids(nodes, elements)
        metrics = compute_metrics(stress_data, element_centroids, sample['defects'])

        if metrics is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return build_row(sim_id, sample, error=True, n_elements=n_elements)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return build_row(sim_id, sample, error=True, n_elements=n_elements)

    dt = time.time() - t0
    row = build_row(sim_id, sample, metrics=metrics)
    row['_time'] = dt
    row['_n_elements'] = n_elements
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return row


def main():
    global _log_fh

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=0,
                        help='Run N sims as smoke test before full batch')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Number of parallel workers (default: {NUM_WORKERS})')
    args = parser.parse_args()

    n_workers = args.workers

    _log_fh = open(LOG_FILE, 'a')

    log("=" * 75)
    log(f"BATCH 20K: CalculiX Composite Plate Simulations ({n_workers} workers)")
    log(f"Output: {OUTPUT_CSV}")
    log(f"Solver: {CCX_EXE}")
    log(f"Random seed: {RANDOM_SEED}")
    log(f"Target: {NUM_SAMPLES} total samples")
    log("=" * 75)

    if not os.path.exists(CCX_EXE):
        log(f"ERROR: Solver not found: {CCX_EXE}")
        log("Install with: sudo apt install calculix-ccx")
        return

    log(f"\nStep 1: Generating all samples (seed={RANDOM_SEED})...")
    t_gen_start = time.time()
    all_samples = generate_all_samples(NUM_SAMPLES, seed=RANDOM_SEED)
    log(f"  Generated {len(all_samples)} samples in {time.time() - t_gen_start:.1f}s")

    counts = {}
    for s in all_samples:
        nd = s['n_defects']
        counts[nd] = counts.get(nd, 0) + 1
    for nd in sorted(counts):
        log(f"    n_defects={nd}: {counts[nd]} samples")

    log("\nStep 2: Generating crack polygons...")
    random.seed(RANDOM_SEED + 7777)
    all_polygons = []
    polygon_failures = set()
    for i, sample in enumerate(all_samples):
        crack_polys = []
        valid = True
        for d in sample['defects']:
            poly = crack_polygon_points(
                d['x'], d['y'], d['half_length'],
                d['width'], d['angle'], d['roughness'])
            if polygon_self_intersects(poly):
                valid = False
                break
            crack_polys.append(poly)
        if valid:
            all_polygons.append(crack_polys)
        else:
            all_polygons.append(None)
            polygon_failures.add(i)
    log(f"  {len(all_samples) - len(polygon_failures)} valid, "
        f"{len(polygon_failures)} self-intersecting")

    completed_sims = load_completed_sims(OUTPUT_CSV)
    write_csv_header(OUTPUT_CSV)
    log(f"\nStep 3: Resume check - {len(completed_sims)} sims already in CSV")

    work_items = []
    for i, sample in enumerate(all_samples):
        sim_id = i + 1
        if sim_id in completed_sims:
            continue
        work_items.append((sim_id, sample, all_polygons[i]))

    if args.test > 0:
        work_items = work_items[:args.test]
        log(f"\n*** TEST MODE: running {args.test} sims only ***")

    to_run = len(work_items)
    log(f"\nStep 4: Running {to_run} simulations with {n_workers} workers...")
    log("-" * 75)

    t_batch_start = time.time()
    n_success = 0
    n_fail = 0
    last_backup = 0

    with Pool(processes=n_workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_single_sim, work_items, chunksize=1)):
            sim_id = row['sim_id']
            is_ok = row['solver_completed'] == 'YES'

            append_csv_row(OUTPUT_CSV, row)

            if is_ok:
                n_success += 1
            else:
                n_fail += 1

            done = n_success + n_fail

            if done % 50 == 0 or done == to_run:
                elapsed = time.time() - t_batch_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (to_run - done) / rate if rate > 0 else 0
                fail_pct = 100.0 * n_fail / done if done > 0 else 0
                log(f"  [{done}/{to_run}] {n_success} OK, {n_fail} ERR ({fail_pct:.1f}%) | "
                    f"{elapsed/60:.1f}min elapsed, ETA {eta/60:.1f}min | "
                    f"{rate*60:.1f} sims/min")

            # bail out early if the whole run is going sideways
            if done >= MIN_SIMS_FOR_FAIL_CHECK and n_fail / done > MAX_FAIL_RATE:
                log(f"\n!!! SAFETY PAUSE: Failure rate {100*n_fail/done:.1f}% exceeds {100*MAX_FAIL_RATE:.0f}% threshold !!!")
                log(f"    {n_fail} failures out of {done} sims")
                log(f"    CSV saved with {done} rows. Re-run to resume from here.")
                backup_path = OUTPUT_CSV.replace('.csv', f'_safepause_{done}.csv')
                shutil.copy2(OUTPUT_CSV, backup_path)
                log(f"    Backup: {backup_path}")
                pool.terminate()
                pool.join()
                _log_fh.close()
                sys.exit(1)

            if done - last_backup >= BACKUP_INTERVAL:
                backup_path = OUTPUT_CSV.replace('.csv', f'_backup_{done}.csv')
                try:
                    shutil.copy2(OUTPUT_CSV, backup_path)
                    log(f"  ** BACKUP saved: {backup_path}")
                except Exception as e:
                    log(f"  ** BACKUP FAILED: {e}")
                last_backup = done

    t_total = time.time() - t_batch_start
    log("\n" + "=" * 75)
    log("BATCH COMPLETE")
    log("=" * 75)
    log(f"  Total samples:    {NUM_SAMPLES}")
    log(f"  Skipped (resume): {len(completed_sims)}")
    log(f"  Ran this session: {n_success + n_fail}")
    log(f"  Successful:       {n_success}")
    log(f"  Failed/Error:     {n_fail}")
    log(f"  Fail rate:        {100.0 * n_fail / (n_success + n_fail):.1f}%" if (n_success + n_fail) > 0 else "  Fail rate:        N/A")
    log(f"  Workers used:     {n_workers}")
    log(f"  Total time:       {t_total:.1f}s ({t_total/60:.1f} min, {t_total/3600:.1f} hrs)")
    if n_success + n_fail > 0:
        log(f"  Avg per sim:      {t_total / (n_success + n_fail):.1f}s (wall)")
        log(f"  Effective speed:  {(n_success + n_fail) / (t_total / 60):.1f} sims/min")
    log(f"  Output CSV:       {OUTPUT_CSV}")
    log("=" * 75)

    final_backup = OUTPUT_CSV.replace('.csv', '_final.csv')
    try:
        shutil.copy2(OUTPUT_CSV, final_backup)
        log(f"  Final backup: {final_backup}")
    except Exception:
        pass

    _log_fh.close()


if __name__ == "__main__":
    main()
