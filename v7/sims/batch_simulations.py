"""
Abaqus batch runner for composite plate with jagged crack defects + progressive damage.

Run headless:  abaqus cae noGUI=batch_simulations.py
Run with GUI:  abaqus cae script=batch_simulations.py
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *
import os
import gc
import math
import random
import time

PLATE_LENGTH = 100.0
PLATE_WIDTH = 50.0

# T300/Epoxy, MPa
E1 = 138000.0
E2 = 8960.0
NU12 = 0.30
G12 = 7100.0
G13 = 7100.0
G23 = 6210.0

XT = 1500.0
XC = 1200.0
YT = 40.0
YC = 246.0
SL = 68.0
ST = 68.0

# Hashin fracture energies, N/mm
GFT = 12.5
GFC = 12.5
GMT = 1.0
GMC = 1.0

# Explicit doesn't support SDEG/HSNFTCRT for composite layup shells (S4R), so
# Standard is used — gives Hashin initiation but not SDEG.
USE_EXPLICIT = False
DENSITY = 1.58e-9           # tonne/mm^3
EXPLICIT_TIME_PERIOD = 0.01
EXPLICIT_TARGET_DT = 1.0e-6

BASE_PLY_ANGLES = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]
N_PLIES = 8

GLOBAL_MESH_SIZE = 3.0
FINE_MESH_SIZE_CRACK = 0.50
CRACK_MESH_SEARCH_BUFFER = 3.0

CRACK_SEG_LEN_MIN = 0.2
CRACK_SEG_LEN_MAX = 0.8
MAX_ANGLE_DEV_DEG = 45.0
MIN_CRACK_WIDTH = 0.15
MIN_POLYGON_SEGMENTS = 12

NUM_SAMPLES = 1000
RANDOM_SEED = 55
MAX_DEFECTS = 5
MAX_PLACEMENT_ATTEMPTS = 200

GLOBAL_RANGES = {
    'pressure_x':      [5.0,   100.0],
    'pressure_y':      [0.0,   100.0],
    'ply_thickness':   [0.10,  0.20],
    'layup_rotation':  [0.0,   90.0],
}

DEFECT_RANGES = {
    'x':              [15.0,  85.0],
    'y':              [10.0,  40.0],
    'half_length':    [4.0,   15.0],
    'width':          [0.15,  0.6],
    'angle':          [0.0,   180.0],
    'roughness':      [0.15,  0.90],
}

OUTPUT_FILE = 'simulation_results_v7.csv'
LOG_FILE = 'v7_batch_log.txt'
SCREENSHOT_DIR = 'v7_sim_captures'
SCREENSHOT_INTERVAL = 25
# True only with 'abaqus cae script=' — printToFile segfaults in noGUI.
SCREENSHOTS_ENABLED = False
ODB_LIST_FILE = 'v7_odb_for_screenshots.txt'


def build_csv_header():
    cols = ['sim_id', 'n_defects']
    for i in range(1, MAX_DEFECTS + 1):
        cols.extend([
            'defect{}_x'.format(i),
            'defect{}_y'.format(i),
            'defect{}_half_length'.format(i),
            'defect{}_width'.format(i),
            'defect{}_angle'.format(i),
            'defect{}_roughness'.format(i),
        ])
    cols.extend([
        'pressure_x', 'pressure_y',
        'ply_thickness', 'layup_rotation',
        'total_thickness',
        'min_inter_defect_dist',
        'max_mises',
        'max_s11', 'min_s11',
        'max_s22', 'min_s22',
        'max_s12',
        'tsai_wu_index',
        'max_hashin_ft', 'max_hashin_fc',
        'max_hashin_mt', 'max_hashin_mc',
    ])
    for i in range(1, MAX_DEFECTS + 1):
        cols.append('max_mises_defect{}'.format(i))
    cols.extend([
        'max_disp',
        'n_elements',
        'failed_tsai_wu',
        'failed_hashin',
        'max_sdeg',
        'n_damaged_elements',
        'solver_completed',
    ])
    return ','.join(cols) + '\n'


CSV_HEADER = build_csv_header()


def log(msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = '[{}] {}'.format(timestamp, msg)
    print(msg)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass


def log_banner():
    divider = '=' * 60
    log(divider)
    log('V7 BATCH RUN LOG')
    log('Started: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    log('Samples: {}'.format(NUM_SAMPLES))
    log('Pressure X range: [{}, {}] MPa'.format(
        GLOBAL_RANGES['pressure_x'][0], GLOBAL_RANGES['pressure_x'][1]))
    log('Mesh: {}mm global, {}mm cracks'.format(
        GLOBAL_MESH_SIZE, FINE_MESH_SIZE_CRACK))
    log('Progressive damage: ON (Hashin evolution, Gft={} Gmt={})'.format(
        GFT, GMT))
    if SCREENSHOTS_ENABLED:
        log('Screenshots: ENABLED every {} sims (printToFile)'.format(
            SCREENSHOT_INTERVAL))
    else:
        log('Screenshots: DISABLED (ODB paths saved to {})'.format(
            ODB_LIST_FILE))
        log('  Run v7_take_screenshots.py in GUI mode afterwards.')
    log(divider)


# session.printToFile() segfaults (signal 11) in noGUI on some Abaqus installs.
# When SCREENSHOTS_ENABLED is False we save ODB paths for later GUI processing.
_screenshot_setup_done = False


def _save_odb_path(odb_path):
    try:
        with open(ODB_LIST_FILE, 'a') as f:
            f.write(os.path.abspath(odb_path) + '\n')
    except Exception:
        pass


def setup_screenshot_quality():
    global _screenshot_setup_done
    if _screenshot_setup_done:
        return
    try:
        session.pngOptions.setValues(imageSize=(4096, 3072))
        session.graphicsOptions.setValues(
            backgroundStyle=SOLID,
            backgroundColor='#FFFFFF')
        vp = session.viewports['Viewport: 1']
        vp.viewportAnnotationOptions.setValues(
            triad=ON, legend=ON, title=ON, state=ON)
        _screenshot_setup_done = True
        log('  Screenshot quality: 4096x3072 PNG, white background')
    except Exception as e:
        log('  WARNING: Could not set screenshot quality: {}'.format(str(e)))


def take_mesh_screenshot(sim_id, part):
    if not SCREENSHOTS_ENABLED:
        return
    try:
        setup_screenshot_quality()
        path = os.path.join(SCREENSHOT_DIR,
                            'mesh_sim_{:04d}'.format(sim_id))
        vp = session.viewports['Viewport: 1']
        vp.setValues(displayedObject=part)
        vp.partDisplay.setValues(mesh=ON)
        vp.view.fitView()
        session.printToFile(fileName=path, format=PNG,
                            canvasObjects=(vp,))
        log('  Screenshot: {}.png'.format(path))
    except Exception as e:
        log('  WARNING: Mesh screenshot failed: {}'.format(str(e)))


def take_deformed_screenshot(sim_id, odb):
    if not SCREENSHOTS_ENABLED:
        return
    try:
        setup_screenshot_quality()
        path = os.path.join(SCREENSHOT_DIR,
                            'deformed_sim_{:04d}'.format(sim_id))
        vp = session.viewports['Viewport: 1']
        vp.setValues(displayedObject=odb)
        vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
        vp.odbDisplay.setPrimaryVariable(
            variableLabel='S', outputPosition=INTEGRATION_POINT,
            refinement=(INVARIANT, 'Mises'))
        vp.odbDisplay.commonOptions.setValues(
            deformationScaling=UNIFORM, uniformScaleFactor=1.0)
        vp.view.fitView()
        session.printToFile(fileName=path, format=PNG,
                            canvasObjects=(vp,))
        log('  Screenshot: {}.png'.format(path))
    except Exception as e:
        log('  WARNING: Deformed screenshot failed: {}'.format(str(e)))


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


def crack_polygon_points(cx, cy, half_length, width, angle_deg,
                         roughness):
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


def validate_crack_bounds(cx, cy, half_length, width, angle_deg,
                          roughness, plate_length, plate_width,
                          margin=2.0):
    max_lateral = width / 2.0 + half_length * 0.3
    t = math.radians(angle_deg)
    dx = abs(half_length * math.cos(t)) + abs(max_lateral * math.sin(t))
    dy = abs(half_length * math.sin(t)) + abs(max_lateral * math.cos(t))
    if cx - dx < margin:
        return False
    if cx + dx > plate_length - margin:
        return False
    if cy - dy < margin:
        return False
    if cy + dy > plate_width - margin:
        return False
    return True


def overlaps_existing(new_defect, existing_defects, margin=2.0):
    for d in existing_defects:
        dist = math.sqrt(
            (new_defect['x'] - d['x'])**2 +
            (new_defect['y'] - d['y'])**2)
        min_dist = (new_defect['half_length'] +
                    d['half_length'] + margin)
        if dist < min_dist:
            return True
    return False


def compute_min_inter_dist(defects):
    n = len(defects)
    if n < 2:
        return 0.0
    min_d = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(
                (defects[i]['x'] - defects[j]['x'])**2 +
                (defects[i]['y'] - defects[j]['y'])**2)
            if d < min_d:
                min_d = d
    return min_d


def get_crack_edges(part, cx, cy, half_length, buffer):
    search_radius = half_length + buffer
    edge_list = []
    for i in range(len(part.edges)):
        edge = part.edges[i]
        pt = edge.pointOn[0]
        dx = pt[0] - cx
        dy = pt[1] - cy
        dist_2d = math.sqrt(dx**2 + dy**2)
        if dist_2d < search_radius:
            edge_list.append(part.edges[i:i + 1])
    if not edge_list:
        return None
    combined = edge_list[0]
    for e in edge_list[1:]:
        combined = combined + e
    return combined


def compute_tsai_wu(s11, s22, s12):
    F1 = 1.0 / XT - 1.0 / XC
    F2 = 1.0 / YT - 1.0 / YC
    F11 = 1.0 / (XT * XC)
    F22 = 1.0 / (YT * YC)
    F66 = 1.0 / (SL * SL)
    F12 = -0.5 * math.sqrt(F11 * F22)
    tw = (F1 * s11 + F2 * s22 +
          F11 * s11**2 + F22 * s22**2 + F66 * s12**2 +
          2.0 * F12 * s11 * s22)
    return tw


def load_completed_ids(filepath):
    """Read CSV, return set of successful sim_ids; strip ERROR rows so they retry."""
    completed = set()
    if not os.path.exists(filepath):
        return completed
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return completed
    header = lines[0]
    good_rows = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) == 0 or not line.strip():
            continue
        if 'ERROR' in line:
            continue
        try:
            completed.add(int(parts[0]))
            good_rows.append(line)
        except ValueError:
            pass
    with open(filepath, 'w') as f:
        f.write(header)
        for row in good_rows:
            f.write(row)
    return completed


def check_solver_completed(job_name):
    try:
        with open(job_name + '.sta', 'r') as sta_f:
            sta_content = sta_f.read().upper()
            if 'SUCCESSFULLY' in sta_content:
                return True
            if 'HAS BEEN COMPLETED' in sta_content:
                return True
    except Exception:
        pass
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
                    PLATE_LENGTH, PLATE_WIDTH):
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
    global_samples = latin_hypercube_sample(
        GLOBAL_RANGES, n_total, seed=seed)

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
                    log('  WARNING: Could not place {} cracks '
                        'after {} attempts. Moving on.'.format(
                            n_def, MAX_SAMPLE_FAILURES))
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

    if placement_failures > 0:
        log('  Placement retries needed: {}'.format(placement_failures))

    random.seed(seed + 999)
    random.shuffle(all_samples)
    return all_samples


def run_single_simulation(sample, sim_id, keep_odb=False,
                          screenshot=False):
    model_name = 'Model_{}'.format(sim_id)
    job_name = 'Job_{:04d}'.format(sim_id)
    job_start = time.time()

    n_def = sample['n_defects']
    defects = sample['defects']
    press_x = sample['pressure_x']
    press_y = sample['pressure_y']
    ply_t = sample['ply_thickness']
    layup_rot = sample['layup_rotation']
    total_thickness = ply_t * N_PLIES
    min_inter_dist = compute_min_inter_dist(defects)

    log('--- JOB {} (sim_id={}) ---'.format(job_name, sim_id))
    log('  n_defects={} Px={:.1f} Py={:.1f} ply_t={:.3f} '
        'rot={:.1f}'.format(n_def, press_x, press_y, ply_t, layup_rot))
    for di, d in enumerate(defects):
        log('  Crack {}: pos=({:.1f},{:.1f}) L={:.1f} w={:.2f} '
            'ang={:.0f} rough={:.2f}'.format(
                di + 1, d['x'], d['y'], d['half_length'],
                d['width'], d['angle'], d['roughness']))

    n_elements = 0

    try:
        if model_name in mdb.models.keys():
            del mdb.models[model_name]

        model = mdb.Model(name=model_name)

        sketch = model.ConstrainedSketch(
            name='plateSketch', sheetSize=200.0)
        sketch.rectangle(
            point1=(0.0, 0.0),
            point2=(PLATE_LENGTH, PLATE_WIDTH))

        for d in defects:
            pts = crack_polygon_points(
                d['x'], d['y'],
                d['half_length'], d['width'],
                d['angle'], d['roughness'])
            n_seg = len(pts)
            for idx in range(n_seg):
                p1 = pts[idx]
                p2 = pts[(idx + 1) % n_seg]
                sketch.Line(point1=p1, point2=p2)

        part = model.Part(
            name='Plate', dimensionality=THREE_D,
            type=DEFORMABLE_BODY)
        part.BaseShell(sketch=sketch)

        n_faces = len(part.faces)
        if n_faces < 1:
            raise Exception('zero faces — sketch produced no geometry')
        log('  Geometry: {} face(s)'.format(n_faces))

        material = model.Material(name='CFRP')
        material.Density(table=((DENSITY,),))
        material.Elastic(
            type=LAMINA,
            table=((E1, E2, NU12, G12, G13, G23),))

        material.HashinDamageInitiation(
            table=((XT, XC, YT, YC, SL, ST),))

        if USE_EXPLICIT:
            try:
                material.hashinDamageInitiation.DamageEvolution(
                    type=ENERGY,
                    softening=LINEAR,
                    table=((GFT, GFC, GMT, GMC),))
                log('  DamageEvolution: created OK')
            except Exception as e:
                log('  CRITICAL: DamageEvolution FAILED: {}'.format(str(e)))
                log('  SDEG will NOT be available in this simulation!')

        log('  Material: CFRP + Hashin{}'.format(
            ' + damage evolution' if USE_EXPLICIT else ' initiation only (Standard)'))

        region = part.Set(faces=part.faces[:], name='AllFaces')
        compositeLayup = part.CompositeLayup(
            name='CompositePlate',
            description='CFRP quasi-isotropic layup',
            elementType=SHELL,
            offsetType=MIDDLE_SURFACE,
            symmetric=False,
            thicknessAssignment=FROM_SECTION)

        compositeLayup.Section(
            preIntegrate=OFF,
            integrationRule=SIMPSON,
            poissonDefinition=DEFAULT,
            thicknessModulus=None,
            temperature=GRADIENT,
            useDensity=OFF)

        compositeLayup.ReferenceOrientation(
            orientationType=GLOBAL,
            localCsys=None,
            additionalRotationType=ROTATION_ANGLE,
            additionalRotationField='',
            angle=layup_rot,
            axis=AXIS_3)

        for i, base_angle in enumerate(BASE_PLY_ANGLES):
            compositeLayup.CompositePly(
                suppressed=False,
                plyName='Ply-{}'.format(i + 1),
                region=region,
                material='CFRP',
                thicknessType=SPECIFY_THICKNESS,
                thickness=ply_t,
                orientationType=SPECIFY_ORIENT,
                orientationValue=base_angle,
                additionalRotationType=ROTATION_NONE,
                additionalRotationField='',
                axis=AXIS_3,
                angle=0.0,
                numIntPoints=3)

        assembly = model.rootAssembly
        assembly.DatumCsysByDefault(CARTESIAN)
        instance = assembly.Instance(
            name='PlateInstance', part=part, dependent=ON)

        if USE_EXPLICIT:
            model.ExplicitDynamicsStep(
                name='LoadStep', previous='Initial',
                timePeriod=EXPLICIT_TIME_PERIOD, nlgeom=ON)
            model.steps['LoadStep'].setValues(
                massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 0,
                              EXPLICIT_TARGET_DT, BELOW_MIN, 0, 0,
                              0.0, 0.0, 0, None),))
            model.SmoothStepAmplitude(name='Ramp', timeSpan=STEP,
                data=((0.0, 0.0), (1.0, 1.0)))
        else:
            model.StaticStep(name='LoadStep', previous='Initial',
                             initialInc=0.1, maxInc=1.0)

        model.fieldOutputRequests['F-Output-1'].setValues(
            variables=('S', 'U', 'RF'), numIntervals=1)
        if USE_EXPLICIT:
            model.FieldOutputRequest(
                name='F-Output-Damage',
                createStepName='LoadStep',
                variables=('SDEG', 'STATUS',
                           'HSNFTCRT', 'HSNFCCRT',
                           'HSNMTCRT', 'HSNMCCRT'),
                numIntervals=1)
        else:
            model.FieldOutputRequest(
                name='F-Output-Damage',
                createStepName='LoadStep',
                variables=('HSNFTCRT', 'HSNFCCRT',
                           'HSNMTCRT', 'HSNMCCRT'),
                numIntervals=1)

        tol = 0.1

        left_edges = instance.edges.getByBoundingBox(
            xMin=-tol, xMax=tol,
            yMin=-tol, yMax=PLATE_WIDTH + tol,
            zMin=-tol, zMax=tol)
        right_edges = instance.edges.getByBoundingBox(
            xMin=PLATE_LENGTH - tol, xMax=PLATE_LENGTH + tol,
            yMin=-tol, yMax=PLATE_WIDTH + tol,
            zMin=-tol, zMax=tol)

        left_region = assembly.Set(
            edges=left_edges, name='LeftEdge')
        model.DisplacementBC(
            name='RollerLeft_U1', createStepName='Initial',
            region=left_region,
            u1=0.0, u2=UNSET, u3=UNSET,
            ur1=UNSET, ur2=UNSET, ur3=UNSET)

        corner_vertex = instance.vertices.getByBoundingBox(
            xMin=-tol, xMax=tol,
            yMin=-tol, yMax=tol,
            zMin=-tol, zMax=tol)
        corner_region = assembly.Set(
            vertices=corner_vertex, name='CornerPin')
        model.DisplacementBC(
            name='CornerPin_U2U3', createStepName='Initial',
            region=corner_region,
            u1=UNSET, u2=0.0, u3=0.0,
            ur1=UNSET, ur2=UNSET, ur3=UNSET)

        load_kw = {'amplitude': 'Ramp'} if USE_EXPLICIT else {}
        right_surface = assembly.Surface(
            side1Edges=right_edges, name='RightEdge')
        force_x = press_x * total_thickness
        model.ShellEdgeLoad(
            name='TensionX',
            createStepName='LoadStep',
            region=right_surface,
            magnitude=force_x,
            directionVector=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            distributionType=UNIFORM,
            traction=GENERAL,
            **load_kw)

        if press_y > 0.0:
            force_y = press_y * total_thickness

            top_edges = instance.edges.getByBoundingBox(
                xMin=-tol, xMax=PLATE_LENGTH + tol,
                yMin=PLATE_WIDTH - tol, yMax=PLATE_WIDTH + tol,
                zMin=-tol, zMax=tol)
            top_surface = assembly.Surface(
                side1Edges=top_edges, name='TopEdge')
            model.ShellEdgeLoad(
                name='TensionY_Top',
                createStepName='LoadStep',
                region=top_surface,
                magnitude=force_y,
                directionVector=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                distributionType=UNIFORM,
                traction=GENERAL,
                **load_kw)

            bottom_edges = instance.edges.getByBoundingBox(
                xMin=-tol, xMax=PLATE_LENGTH + tol,
                yMin=-tol, yMax=tol,
                zMin=-tol, zMax=tol)
            bottom_surface = assembly.Surface(
                side1Edges=bottom_edges, name='BottomEdge')
            model.ShellEdgeLoad(
                name='TensionY_Bottom',
                createStepName='LoadStep',
                region=bottom_surface,
                magnitude=force_y,
                directionVector=((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
                distributionType=UNIFORM,
                traction=GENERAL,
                **load_kw)

        part.seedPart(
            size=GLOBAL_MESH_SIZE,
            deviationFactor=0.1,
            minSizeFactor=0.1)

        for d in defects:
            edges = get_crack_edges(
                part, d['x'], d['y'],
                d['half_length'], CRACK_MESH_SEARCH_BUFFER)
            if edges is not None:
                part.seedEdgeBySize(
                    edges=edges, size=FINE_MESH_SIZE_CRACK,
                    deviationFactor=0.1, constraint=FINER)

        if USE_EXPLICIT:
            elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=EXPLICIT,
                hourglassControl=ENHANCED)
            elemType2 = mesh.ElemType(elemCode=S3, elemLibrary=EXPLICIT)
        else:
            elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD)
            elemType2 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
        all_faces = part.faces[:]
        part.setMeshControls(regions=all_faces, technique=FREE)
        part.setElementType(
            regions=(all_faces,),
            elemTypes=(elemType1, elemType2))

        part.generateMesh()
        n_elements = len(part.elements)
        log('  Mesh: {} elements'.format(n_elements))
        if n_elements == 0:
            raise Exception(
                'Mesh generation produced 0 elements — geometry too '
                'complex for mesher (n_defects={})'.format(n_def))

        if screenshot:
            take_mesh_screenshot(sim_id, part)

        mdb.Job(name=job_name, model=model_name, type=ANALYSIS)
        mdb.jobs[job_name].writeInput()

        # sanity-check the .inp has what the solver needs
        inp_path = job_name + '.inp'
        try:
            with open(inp_path, 'r') as inp_f:
                inp_text = inp_f.read()
            if USE_EXPLICIT:
                if '*Damage Evolution' not in inp_text:
                    log('  CRITICAL: *Damage Evolution NOT found in {}!'.format(
                        inp_path))
                    log('  The DamageEvolution API call may have silently failed.')
                else:
                    log('  Verified: *Damage Evolution present in .inp')
                if 'SDEG' not in inp_text:
                    log('  CRITICAL: SDEG not in .inp field output request!')
                    log('  Abaqus will not write SDEG to the ODB.')
                else:
                    log('  Verified: SDEG requested in .inp')
                if '*Dynamic' not in inp_text:
                    log('  CRITICAL: *Dynamic not found in .inp!')
                else:
                    log('  Verified: *Dynamic present in .inp')
            else:
                if '*Static' not in inp_text:
                    log('  CRITICAL: *Static not found in .inp!')
                else:
                    log('  Verified: *Static present in .inp')
        except Exception as e:
            log('  WARNING: Could not verify .inp file: {}'.format(str(e)))

        solver_cmd = 'abaqus job={} interactive'.format(job_name)
        log('  Solver: submitting...')
        solver_rc = os.system(solver_cmd)

        solver_completed = check_solver_completed(job_name)
        odb_exists = os.path.exists(job_name + '.odb')
        log('  Solver: rc={} completed={} odb={}'.format(
            solver_rc,
            'YES' if solver_completed else 'NO',
            'YES' if odb_exists else 'NO'))

        if not solver_completed:
            for diag_ext in ['.msg', '.sta', '.dat']:
                diag_path = job_name + diag_ext
                if os.path.exists(diag_path):
                    try:
                        with open(diag_path, 'r') as df:
                            diag_lines = df.readlines()
                        log('  --- {} (last 20 lines) ---'.format(diag_ext))
                        for dl in diag_lines[-20:]:
                            log('  | ' + dl.rstrip())
                        log('  --- end {} ---'.format(diag_ext))
                    except Exception:
                        pass

        if not solver_completed and not odb_exists:
            raise Exception('Job did not complete (rc={}) and no ODB'.format(
                solver_rc))

        if not solver_completed and odb_exists:
            log('  INFO: solver stopped early — extracting partial results')

        # pre-init so partial extraction still returns safe defaults
        max_mises = 0.0
        max_s11 = -1e20
        min_s11 = 1e20
        max_s22 = -1e20
        min_s22 = 1e20
        max_s12 = 0.0
        max_tw = 0.0
        max_mises_per_defect = [0.0] * MAX_DEFECTS
        max_hashin_ft = 0.0
        max_hashin_fc = 0.0
        max_hashin_mt = 0.0
        max_hashin_mc = 0.0
        max_sdeg = 0.0
        n_damaged_elements = 0
        max_disp = 0.0

        odb = session.openOdb(name=job_name + '.odb')
        try:
            if not SCREENSHOTS_ENABLED:
                _save_odb_path(job_name + '.odb')
            if screenshot:
                take_deformed_screenshot(sim_id, odb)

            step = odb.steps['LoadStep']
            n_frames = len(step.frames)
            log('  ODB: LoadStep has {} frame(s)'.format(n_frames))
            if n_frames == 0:
                raise Exception(
                    'LoadStep has 0 frames — solver produced no output. '
                    'Check .msg file above for errors.')
            frame = step.frames[-1]

            stress_field = frame.fieldOutputs['S']
            stress_at_nodes = stress_field.getSubset(
                position=ELEMENT_NODAL)

            inst_nodes = odb.rootAssembly.instances[
                'PLATEINSTANCE'].nodes
            node_coords = {}
            for nd in inst_nodes:
                node_coords[nd.label] = (nd.coordinates[0],
                                         nd.coordinates[1])

            max_mises = 0.0
            max_s11 = -1e20
            min_s11 = 1e20
            max_s22 = -1e20
            min_s22 = 1e20
            max_s12 = 0.0
            max_tw = 0.0
            max_mises_per_defect = [0.0] * MAX_DEFECTS

            for value in stress_at_nodes.values:
                mises = value.mises
                nl = value.nodeLabel
                if nl <= 0:
                    continue

                s11 = value.data[0]
                s22 = value.data[1]
                s12 = value.data[2]

                if mises > max_mises:
                    max_mises = mises
                if s11 > max_s11:
                    max_s11 = s11
                if s11 < min_s11:
                    min_s11 = s11
                if s22 > max_s22:
                    max_s22 = s22
                if s22 < min_s22:
                    min_s22 = s22
                if abs(s12) > max_s12:
                    max_s12 = abs(s12)

                tw = compute_tsai_wu(s11, s22, s12)
                if tw > max_tw:
                    max_tw = tw

                if nl not in node_coords:
                    continue
                nx, ny = node_coords[nl]
                for di in range(n_def):
                    d = defects[di]
                    search_r = d['half_length'] * 2.0 + 5.0
                    dd = math.sqrt(
                        (nx - d['x'])**2 + (ny - d['y'])**2)
                    if dd < search_r:
                        if mises > max_mises_per_defect[di]:
                            max_mises_per_defect[di] = mises

            max_hashin_ft = 0.0
            max_hashin_fc = 0.0
            max_hashin_mt = 0.0
            max_hashin_mc = 0.0

            hashin_keys = [
                ('HSNFTCRT', 'ft'),
                ('HSNFCCRT', 'fc'),
                ('HSNMTCRT', 'mt'),
                ('HSNMCCRT', 'mc'),
            ]
            for key, mode in hashin_keys:
                try:
                    field = frame.fieldOutputs[key]
                    for value in field.values:
                        raw = value.data
                        val = raw[0] if hasattr(raw, '__len__') else raw
                        if mode == 'ft' and val > max_hashin_ft:
                            max_hashin_ft = val
                        elif mode == 'fc' and val > max_hashin_fc:
                            max_hashin_fc = val
                        elif mode == 'mt' and val > max_hashin_mt:
                            max_hashin_mt = val
                        elif mode == 'mc' and val > max_hashin_mc:
                            max_hashin_mc = val
                except KeyError:
                    log('  WARNING: {} not in ODB'.format(key))

            max_sdeg = 0.0
            n_damaged_elements = 0
            try:
                sdeg_field = frame.fieldOutputs['SDEG']
                damaged_elem_labels = set()
                for value in sdeg_field.values:
                    raw = value.data
                    val = raw[0] if hasattr(raw, '__len__') else raw
                    if val > max_sdeg:
                        max_sdeg = val
                    if val >= 0.99:
                        damaged_elem_labels.add(value.elementLabel)
                n_damaged_elements = len(damaged_elem_labels)
            except KeyError:
                log('  WARNING: SDEG not in ODB')

            disp_field = frame.fieldOutputs['U']
            max_disp = 0.0
            for value in disp_field.values:
                mag = math.sqrt(
                    value.data[0]**2 +
                    value.data[1]**2 +
                    value.data[2]**2)
                if mag > max_disp:
                    max_disp = mag

            # corner pin should carry near-zero reaction
            try:
                rf_field = frame.fieldOutputs['RF']
                corner_set = odb.rootAssembly.nodeSets['CORNERPIN']
                corner_rf = rf_field.getSubset(region=corner_set)
                for value in corner_rf.values:
                    RF2 = value.data[1]
                    RF3 = value.data[2]
                    if abs(RF2) > 1.0 or abs(RF3) > 1.0:
                        log('  WARNING: Corner pin RF > 1 N '
                            '(RF2={:.2e} RF3={:.2e})'.format(RF2, RF3))
            except Exception:
                pass

        finally:
            odb.close()

        failed_tw = 1 if max_tw >= 1.0 else 0
        failed_hashin = 1 if max(
            max_hashin_ft, max_hashin_fc,
            max_hashin_mt, max_hashin_mc) >= 1.0 else 0

        del mdb.models[model_name]
        if job_name in mdb.jobs.keys():
            del mdb.jobs[job_name]

        extensions = ['.inp', '.lck', '.res', '.prt', '.sim',
                      '.sta', '.msg', '.dat', '.com', '.ipm',
                      '.log', '.023', '.SMABulk', '.cid']
        if not keep_odb:
            extensions.append('.odb')
        for ext in extensions:
            try:
                os.remove(job_name + ext)
            except Exception:
                pass
        try:
            import shutil
            shutil.rmtree(job_name + '.simdir', ignore_errors=True)
        except Exception:
            pass

        gc.collect()

        elapsed = time.time() - job_start
        log('  Results: mises={:.1f} TW={:.3f} H_ft={:.3f} H_mt={:.3f} '
            'SDEG={:.4f} dmg_el={} disp={:.4f}'.format(
                max_mises, max_tw, max_hashin_ft, max_hashin_mt,
                max_sdeg, n_damaged_elements, max_disp))
        log('  Failure: TW={} Hashin={} solver={}'.format(
            'YES' if failed_tw else 'NO',
            'YES' if failed_hashin else 'NO',
            'OK' if solver_completed else 'STOPPED'))
        log('  Duration: {:.1f}s'.format(elapsed))

        result = {
            'sim_id': sim_id,
            'n_defects': n_def,
            'pressure_x': press_x,
            'pressure_y': press_y,
            'ply_thickness': ply_t,
            'layup_rotation': layup_rot,
            'total_thickness': total_thickness,
            'min_inter_defect_dist': min_inter_dist,
            'max_mises': max_mises,
            'max_s11': max_s11,
            'min_s11': min_s11,
            'max_s22': max_s22,
            'min_s22': min_s22,
            'max_s12': max_s12,
            'tsai_wu_index': max_tw,
            'max_hashin_ft': max_hashin_ft,
            'max_hashin_fc': max_hashin_fc,
            'max_hashin_mt': max_hashin_mt,
            'max_hashin_mc': max_hashin_mc,
            'max_mises_per_defect': max_mises_per_defect,
            'max_disp': max_disp,
            'n_elements': n_elements,
            'failed_tsai_wu': failed_tw,
            'failed_hashin': failed_hashin,
            'max_sdeg': max_sdeg,
            'n_damaged_elements': n_damaged_elements,
            'solver_completed': 1 if solver_completed else 0,
        }

        for i in range(MAX_DEFECTS):
            if i < n_def:
                d = defects[i]
                result['defect{}_x'.format(i + 1)] = d['x']
                result['defect{}_y'.format(i + 1)] = d['y']
                result['defect{}_half_length'.format(i + 1)] = d['half_length']
                result['defect{}_width'.format(i + 1)] = d['width']
                result['defect{}_angle'.format(i + 1)] = d['angle']
                result['defect{}_roughness'.format(i + 1)] = d['roughness']
            else:
                result['defect{}_x'.format(i + 1)] = 0.0
                result['defect{}_y'.format(i + 1)] = 0.0
                result['defect{}_half_length'.format(i + 1)] = 0.0
                result['defect{}_width'.format(i + 1)] = 0.0
                result['defect{}_angle'.format(i + 1)] = 0.0
                result['defect{}_roughness'.format(i + 1)] = 0.0

        return result

    except Exception as e:
        elapsed = time.time() - job_start
        log('  ERROR: {}'.format(str(e)))
        log('  Duration: {:.1f}s'.format(elapsed))
        try:
            if model_name in mdb.models.keys():
                del mdb.models[model_name]
        except Exception:
            pass
        extensions = ['.inp', '.odb', '.lck', '.res', '.prt', '.sim',
                      '.sta', '.msg', '.dat', '.com', '.ipm',
                      '.log', '.023', '.SMABulk', '.cid']
        for ext in extensions:
            try:
                os.remove(job_name + ext)
            except Exception:
                pass
        gc.collect()
        return None


def write_csv_row(filepath, result):
    vals = [
        result['sim_id'],
        result['n_defects'],
    ]
    for i in range(1, MAX_DEFECTS + 1):
        vals.extend([
            result['defect{}_x'.format(i)],
            result['defect{}_y'.format(i)],
            result['defect{}_half_length'.format(i)],
            result['defect{}_width'.format(i)],
            result['defect{}_angle'.format(i)],
            result['defect{}_roughness'.format(i)],
        ])
    vals.extend([
        result['pressure_x'],
        result['pressure_y'],
        result['ply_thickness'],
        result['layup_rotation'],
        result['total_thickness'],
        result['min_inter_defect_dist'],
        result['max_mises'],
        result['max_s11'],
        result['min_s11'],
        result['max_s22'],
        result['min_s22'],
        result['max_s12'],
        result['tsai_wu_index'],
        result['max_hashin_ft'],
        result['max_hashin_fc'],
        result['max_hashin_mt'],
        result['max_hashin_mc'],
    ])
    for i in range(MAX_DEFECTS):
        vals.append(result['max_mises_per_defect'][i])
    vals.extend([
        result['max_disp'],
        result['n_elements'],
        result['failed_tsai_wu'],
        result['failed_hashin'],
        result['max_sdeg'],
        result['n_damaged_elements'],
        result['solver_completed'],
    ])
    try:
        with open(filepath, 'a') as f:
            f.write(','.join([str(v) for v in vals]) + '\n')
    except Exception as e:
        log('  CRITICAL: CSV write failed: {}'.format(str(e)))


def write_error_row(filepath, sim_id, sample, error_msg=''):
    n_def = sample['n_defects']
    defects = sample['defects']
    vals = [sim_id, n_def]
    for i in range(MAX_DEFECTS):
        if i < n_def:
            d = defects[i]
            vals.extend([d['x'], d['y'], d['half_length'],
                         d['width'], d['angle'], d['roughness']])
        else:
            vals.extend([0, 0, 0, 0, 0, 0])
    vals.extend([
        sample['pressure_x'], sample['pressure_y'],
        sample['ply_thickness'], sample['layup_rotation'],
        sample['ply_thickness'] * N_PLIES,
    ])
    vals.append(compute_min_inter_dist(defects))
    # 11 stress/hashin + MAX_DEFECTS per-defect mises + 7 trailing = 23
    n_output_cols = 11 + MAX_DEFECTS + 4 + 3
    vals.extend(['ERROR'] * n_output_cols)
    try:
        with open(filepath, 'a') as f:
            f.write(','.join([str(v) for v in vals]) + '\n')
    except Exception as e:
        log('  CRITICAL: Error row CSV write failed: {}'.format(str(e)))


def main():
    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)

    log_banner()

    log('Generating {} samples (stratified by crack count)...'.format(
        NUM_SAMPLES))
    all_samples = generate_all_samples(NUM_SAMPLES, seed=RANDOM_SEED)
    total_sims = len(all_samples)
    log('Total valid samples: {}'.format(total_sims))

    for nd in range(1, MAX_DEFECTS + 1):
        count = sum([1 for s in all_samples if s['n_defects'] == nd])
        log('  n_defects={}: {} samples'.format(nd, count))

    completed_ids = load_completed_ids(OUTPUT_FILE)
    if completed_ids:
        log('RESUMING: {} already completed'.format(len(completed_ids)))

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w') as f:
            f.write(CSV_HEADER)

    log('Output: {}'.format(OUTPUT_FILE))
    log('=' * 60)

    successful = 0
    failed_count = 0
    skipped = 0
    start_time = time.time()

    for i, sample in enumerate(all_samples):
        sim_id = i + 1

        if sim_id in completed_ids:
            skipped += 1
            continue

        elapsed = time.time() - start_time
        done = successful + failed_count
        if done > 0:
            avg_time = elapsed / done
            remaining = avg_time * (total_sims - skipped - done)
            eta_str = ' ETA {:.0f}min'.format(remaining / 60.0)
        else:
            eta_str = ''

        log('[{}/{}]{}'.format(sim_id, total_sims, eta_str))

        do_screenshot = (sim_id == 1 or sim_id % SCREENSHOT_INTERVAL == 0)

        result = run_single_simulation(
            sample, sim_id,
            keep_odb=(not SCREENSHOTS_ENABLED and do_screenshot),
            screenshot=do_screenshot)

        if result is not None:
            write_csv_row(OUTPUT_FILE, result)
            log('  CSV: row written (SUCCESS)')
            successful += 1
        else:
            write_error_row(OUTPUT_FILE, sim_id, sample)
            log('  CSV: error row written (FAILED)')
            failed_count += 1

    total_time = time.time() - start_time
    log('')
    log('=' * 60)
    log('V7 BATCH COMPLETE')
    log('=' * 60)
    log('Total time: {:.1f} minutes'.format(total_time / 60.0))
    log('Successful: {}'.format(successful))
    log('Failed: {}'.format(failed_count))
    if skipped > 0:
        log('Skipped (resumed): {}'.format(skipped))
    if successful + failed_count > 0:
        log('Success rate: {:.1f}%'.format(
            100.0 * successful / (successful + failed_count)))
    log('Results: {}'.format(OUTPUT_FILE))
    log('Log: {}'.format(LOG_FILE))
    log('Screenshots: {}/'.format(SCREENSHOT_DIR))
    log('=' * 60)


if __name__ == '__main__':
    main()
