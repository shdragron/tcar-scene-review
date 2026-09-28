"""nuScenes-format sample viewer (6 cameras + LIDAR_TOP), stdlib only.

    python tools/sample_viewer.py [--dataroot D:/tcar_nuscenes] [--version v1.0-trainval] [--port 8765]

Then open http://localhost:8765 in a browser (simple viewer); http://localhost:8765/full is the earlier
detailed viewer (distribution, risk, labels, selection panels).
"""
import argparse
import bisect
import json
import math
import mimetypes
import os
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

try:                                        # optional: faster tracks and the compact /lidar/ endpoint
    import numpy as np
except ImportError:
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def ll2utm(lat, lon, zone=52):
    """WGS84 lat/lon (deg) -> UTM easting/northing (m) and grid convergence (rad)."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = math.radians(lat), math.radians(lon)
    dlam = lam - math.radians((zone - 1) * 6 - 180 + 3)
    s, c = math.sin(phi), math.cos(phi)
    N = a / math.sqrt(1 - e2 * s * s)
    T, C, A = math.tan(phi) ** 2, ep2 * c * c, c * dlam
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    y = k0 * (M + N * math.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                       + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y, math.atan(math.tan(dlam) * s)


def _ll2utm_np(np, lat, lon, zone=52):
    """ll2utm on arrays (same formulas); ~10x faster than calling ll2utm per 100 Hz row."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = np.radians(lat), np.radians(lon)
    dlam = lam - math.radians((zone - 1) * 6 - 180 + 3)
    s, c = np.sin(phi), np.cos(phi)
    N = a / np.sqrt(1 - e2 * s * s)
    T, C, A = np.tan(phi) ** 2, ep2 * c * c, c * dlam
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    y = k0 * (M + N * np.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                     + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y, np.arctan(np.tan(dlam) * s)


def load_canbus(dataroot, scene_name):
    """Per-scene INS track from can_bus/<scene>_pose.json.

    Position comes from inspva lat/lon (100 Hz) rather than odom `pos`, which drops to
    ~1 Hz in most scenes. Yaw is rotated from true north to the UTM grid.
    """
    path = os.path.join(dataroot, 'can_bus', scene_name + '_pose.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        rows = json.load(f)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and rows:                   # vectorised path (numpy is optional for the viewer)
        lat, lon = np.array([r['lat'] for r in rows]), np.array([r['lon'] for r in rows])
        x, y, gamma = _ll2utm_np(np, lat, lon)
        q = np.array([r['orientation'] for r in rows])
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)) + gamma
        return {'t': [r['utime'] for r in rows], 'x': x.tolist(), 'y': y.tolist(), 'yaw': yaw.tolist(),
                'speed': [math.hypot(r['vel'][0], r['vel'][1]) for r in rows],
                'ax': [r['accel'][0] for r in rows], 'ay': [r['accel'][1] for r in rows],
                'wz': [r['rotation_rate'][2] for r in rows], 'ins': [r['ins_status'] for r in rows],
                'lat': lat.tolist(), 'lon': lon.tolist(), 'h': [r['height'] for r in rows]}
    tr = {k: [] for k in ('t', 'x', 'y', 'yaw', 'speed', 'ax', 'ay', 'wz', 'ins', 'lat', 'lon', 'h')}
    for r in rows:
        x, y, gamma = ll2utm(r['lat'], r['lon'])
        w, qx, qy, qz = r['orientation']
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) + gamma
        tr['t'].append(r['utime']); tr['x'].append(x); tr['y'].append(y); tr['yaw'].append(yaw)
        tr['speed'].append(math.hypot(r['vel'][0], r['vel'][1]))
        tr['ax'].append(r['accel'][0]); tr['ay'].append(r['accel'][1]); tr['wz'].append(r['rotation_rate'][2])
        tr['ins'].append(r['ins_status']); tr['lat'].append(r['lat']); tr['lon'].append(r['lon']); tr['h'].append(r['height'])
    return tr


def load_db(dataroot, version, ego_poses=True):
    """ego_poses=False skips ego_pose.json (41 MB): only the viewer shows stored ego poses, the analysis
    tools use the INS track."""
    t0 = time.time()
    d = os.path.join(dataroot, version)

    def load(name):
        with open(os.path.join(d, name + '.json'), encoding='utf-8') as f:
            return json.load(f)

    logs = {l['token']: l for l in load('log')}
    sensors = {s['token']: s for s in load('sensor')}
    calib = {c['token']: c for c in load('calibrated_sensor')}
    samples = {s['token']: s for s in load('sample')}
    scenes = load('scene')

    # Key frames: one sample_data per channel per sample. All frames (sweeps too) are kept per
    # scene/channel as (timestamp, filename) so the viewer can play full-rate video.
    scene_name = {s['token']: s['name'] for s in scenes}
    by_sample, frames = {}, {}
    for sd in load('sample_data'):
        ch = sensors[calib[sd['calibrated_sensor_token']]['sensor_token']]['channel']
        sn = scene_name.get(samples[sd['sample_token']]['scene_token'])
        frames.setdefault(sn, {}).setdefault(ch, []).append((sd['timestamp'], sd['filename']))
        if sd['is_key_frame']:
            by_sample.setdefault(sd['sample_token'], {})[ch] = sd
    for chans in frames.values():
        for lst in chans.values():
            lst.sort()

    ego = {}
    if ego_poses:
        needed = {sd['ego_pose_token'] for chans in by_sample.values() for sd in chans.values()}
        ego = {e['token']: e for e in load('ego_pose') if e['token'] in needed}

    scene_list, tracks, scene_of = [], {}, {}
    for sc in sorted(scenes, key=lambda s: s['name']):
        toks, tok = [], sc['first_sample_token']
        while tok:
            toks.append(tok)
            scene_of[tok] = sc['name']
            tok = samples[tok]['next']
        tr = load_canbus(dataroot, sc['name'])
        tracks[sc['name']] = tr
        scene_list.append({
            'token': sc['token'], 'name': sc['name'], 'description': sc['description'],
            'log': logs[sc['log_token']]['logfile'], 'samples': toks,
            'sample_ts': [samples[t]['timestamp'] for t in toks], 'tr': tr,
        })

    # Route for the map: 10 Hz, in a local frame (origin = min UTM over all scenes) to keep numbers small.
    xs = [v for tr in tracks.values() if tr for v in tr['x']]
    ys = [v for tr in tracks.values() if tr for v in tr['y']]
    origin = [math.floor(min(xs)), math.floor(min(ys))] if xs else [0, 0]
    for s in scene_list:
        tr = s.pop('tr')
        s['route'] = [] if not tr else [
            [round(tr['x'][i] - origin[0], 2), round(tr['y'][i] - origin[1], 2), tr['t'][i], round(tr['speed'][i] * 3.6, 2)]
            for i in range(0, len(tr['t']), 10)]

    print(f'[viewer] loaded {len(scene_list)} scenes, {len(by_sample)} samples '
          f'in {time.time() - t0:.1f}s', flush=True)
    return {'scenes': scene_list, 'samples': samples, 'by_sample': by_sample, 'frames': frames,
            'ego': ego, 'calib': calib, 'tracks': tracks, 'scene_of': scene_of, 'origin': origin}


def ego_status(db, token, ts, ego_translation):
    tr = db['tracks'].get(db['scene_of'].get(token))
    if not tr:
        return None
    i = bisect.bisect_left(tr['t'], ts)
    if i > 0 and (i == len(tr['t']) or ts - tr['t'][i - 1] < tr['t'][i] - ts):
        i -= 1
    ox, oy = db['origin']
    return {
        'dt_ms': (tr['t'][i] - ts) / 1000.0,
        'x': tr['x'][i] - ox, 'y': tr['y'][i] - oy, 'utm': [tr['x'][i], tr['y'][i]],
        'yaw_deg': math.degrees(tr['yaw'][i]), 'speed_kmh': tr['speed'][i] * 3.6,
        'ax': tr['ax'][i], 'ay': tr['ay'][i], 'wz_dps': math.degrees(tr['wz'][i]),
        'ins_status': tr['ins'][i], 'lat': tr['lat'][i], 'lon': tr['lon'][i], 'height': tr['h'][i],
        # how far the stored ego_pose is from the INS track (large where odom pos froze at 1 Hz)
        'ego_pose_err_m': math.hypot(ego_translation[0] - tr['x'][i], ego_translation[1] - tr['y'][i])
        if ego_translation else None,
    }


def sample_payload(db, token):
    chans = db['by_sample'].get(token)
    if chans is None:
        return None
    ref_ts = chans['LIDAR_TOP']['timestamp'] if 'LIDAR_TOP' in chans else db['samples'][token]['timestamp']
    out = {'token': token, 'timestamp': db['samples'][token]['timestamp'], 'channels': {}}
    for ch, sd in chans.items():
        e = db['ego'][sd['ego_pose_token']]
        c = db['calib'][sd['calibrated_sensor_token']]
        out['channels'][ch] = {
            'token': sd['token'], 'filename': sd['filename'], 'timestamp': sd['timestamp'],
            'dt_ms': (sd['timestamp'] - ref_ts) / 1000.0,
            'ego_translation': e['translation'], 'ego_rotation': e['rotation'],
            'calib_translation': c['translation'], 'calib_rotation': c['rotation'],
            'camera_intrinsic': c['camera_intrinsic'],
        }
    lid = out['channels'].get('LIDAR_TOP')
    out['ego_status'] = ego_status(db, token, ref_ts, lid['ego_translation'] if lid else None)
    return out


LABELS = ('keep', 'must_keep', 'duplicate', '')


def read_json(path, default):
    if not os.path.isfile(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_final(sel_dir):
    """selection/final_selection.json = the rule's proposals (result.json) overridden by the viewer's human
    decisions (human_decisions.json): the list to actually keep / delete. Rewritten on every decision and
    after every curate.py run."""
    res = read_json(os.path.join(sel_dir, 'result.json'), None)
    if not res or res.get('missing'):
        return None
    dec = read_json(os.path.join(sel_dir, 'human_decisions.json'), {'decisions': {}}).get('decisions', {})
    scenes = {}
    for n, s in sorted(res['scenes'].items()):
        rule = 'drop' if s.get('status') == 'remove' else 'keep'
        h = (dec.get(n) or {}).get('decision')
        scenes[n] = {'final': h or rule, 'by': 'human' if h else 'rule', 'rule': rule}
    drop = [n for n, v in scenes.items() if v['final'] == 'drop']
    out = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'rule': {'mode': res.get('mode'), 'created': res.get('created')},
           'counts': {'scenes': len(scenes), 'kept': len(scenes) - len(drop), 'deleted': len(drop),
                      'human_decisions': sum(1 for v in scenes.values() if v['by'] == 'human'),
                      'proposals_not_reviewed': sum(1 for v in scenes.values() if v['rule'] == 'drop' and v['by'] == 'rule')},
           'deleted': drop, 'kept': [n for n, v in scenes.items() if v['final'] == 'keep'], 'scenes': scenes}
    path = os.path.join(sel_dir, 'final_selection.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(path + '.tmp', path)
    return out


def make_handler(db, dataroot, version='v1.0-trainval'):
    root = os.path.realpath(dataroot)
    html_path = os.path.join(HERE, 'sample_viewer.html')
    full_path = os.path.join(HERE, 'sample_viewer_full.html')      # the earlier detailed viewer, at /full
    # scene-selection files: result.json is written by tools/scene_select.py, labels by this viewer
    sel_dir = os.path.join(dataroot, 'selection')
    labels_path = os.path.join(sel_dir, 'human_labels.json')
    pairs_path = os.path.join(sel_dir, 'pair_labels.json')     # human "same situation?" judgements
    decisions_path = os.path.join(sel_dir, 'human_decisions.json')   # 남기기 / 버리기 chosen in the viewer
    result_path = os.path.join(sel_dir, 'result.json')
    scene_names = {s['name'] for s in db['scenes']}
    labels_lock = threading.Lock()

    def update_decision(body):
        scene, decision, via = body.get('scene'), body.get('decision'), body.get('via', 'manual')
        if scene not in scene_names or decision not in ('keep', 'drop', None):
            raise ValueError('bad scene or decision')
        with labels_lock:
            data = read_json(decisions_path, {'version': 1, 'decisions': {}})
            if decision is None:
                data['decisions'].pop(scene, None)                 # back to the automatic decision
            else:
                data['decisions'][scene] = {'decision': decision, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S')}
            os.makedirs(sel_dir, exist_ok=True)
            tmp = decisions_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, decisions_path)
            write_final(sel_dir)
            # human-readable history, one line per decision
            rule = (read_json(result_path, {}).get('scenes', {}).get(scene) or {}).get('status')
            with open(os.path.join(sel_dir, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {scene}  "
                        f"{ {'keep': '남김', 'drop': '버림', None: '취소(규칙 제안으로)'}[decision] }  "
                        f"(규칙: {'삭제 제안' if rule == 'remove' else '남김'} · {'규칙대로 확정' if via == 'rule' else '직접 선택'})\n")
            return data

    apply_lock = threading.Lock()

    def apply_plan():
        import apply_selection
        p = apply_selection.plan(dataroot)
        fin = read_json(os.path.join(sel_dir, 'final_selection.json'), {}).get('scenes', {})
        p['by'] = {n: (fin.get(n) or {}).get('by', 'rule') for n in p['deleted']}
        return p

    def run_apply(body):
        """삭제 확정: move the deleted scenes out, renumber, reload, rerun the rule on the new dataset."""
        if body.get('confirm') is not True:
            raise ValueError('confirm required')
        if not apply_lock.acquire(blocking=False):
            raise ValueError('already running')
        try:
            import subprocess
            import apply_selection
            try:
                with labels_lock:
                    r = apply_selection.apply(dataroot)
            except Exception as e:                                 # partial moves are in <dataroot>/_removed/<stamp>/manifest.json
                return {'ok': False, 'error': f'{type(e).__name__}: {e}',
                        'hint': '중간에 멈췄다면 _removed/ 아래 가장 최근 폴더로 --undo 하면 원래대로 돌아갑니다'}
            fresh = load_db(dataroot, version)                     # the viewer now serves the new dataset
            db.clear(); db.update(fresh)
            scene_names.clear(); scene_names.update(s['name'] for s in db['scenes'])
            cur = subprocess.run([sys.executable, os.path.join(HERE, 'curate.py'), '--dataroot', dataroot],
                                 capture_output=True, text=True, encoding='utf-8', errors='replace',
                                 env=dict(os.environ, PYTHONIOENCODING='utf-8'))
            return {'ok': True, 'backup': r['backup'], 'deleted': r['deleted'], 'rename': r['rename'],
                    'n_after': r['n_after'], 'curate_ok': cur.returncode == 0,
                    'undo': f'python tools/apply_selection.py --undo "{r["backup"]}"'}
        finally:
            apply_lock.release()

    def update_label(body):
        scene, label = body.get('scene'), body.get('label', '')
        dup_of, note = body.get('dup_of') or None, str(body.get('note', ''))[:2000]
        if scene not in scene_names or label not in LABELS:
            raise ValueError('bad scene or label')
        if dup_of is not None and (dup_of not in scene_names or dup_of == scene):
            raise ValueError('bad dup_of')
        with labels_lock:
            data = read_json(labels_path, {'version': 1, 'labels': {}})
            if label or note:
                data['labels'][scene] = {'label': label, 'dup_of': dup_of if label == 'duplicate' else None,
                                         'note': note, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S')}
            else:
                data['labels'].pop(scene, None)
            os.makedirs(sel_dir, exist_ok=True)
            tmp = labels_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, labels_path)
            return data

    def update_pair(body):
        a, b, same = body.get('a'), body.get('b'), body.get('same')
        if a not in scene_names or b not in scene_names or a == b or same not in (True, False, None):
            raise ValueError('bad pair')
        key = '|'.join(sorted((a, b)))
        with labels_lock:
            data = read_json(pairs_path, {'version': 1, 'pairs': {}})
            if same is None:
                data['pairs'].pop(key, None)
            else:
                data['pairs'][key] = {'same': same, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S')}
            os.makedirs(sel_dir, exist_ok=True)
            tmp = pairs_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, pairs_path)
            return data

    class Handler(SimpleHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'          # keep-alive: playback fetches ~70 files/s, one TCP connection each was too many

        def log_message(self, fmt, *args):
            pass

        def send_bytes(self, body, ctype, cache=False):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'max-age=3600' if cache else 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj):
            self.send_bytes(json.dumps(obj).encode('utf-8'), 'application/json')

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path == '/favicon.ico':                 # no icon; answer quietly instead of a 404 in the console
                    self.send_response(204); self.end_headers()
                    return
                if path in ('/', '/index.html', '/full'):
                    # Re-read on every request so HTML edits apply without a restart.
                    with open(full_path if path == '/full' else html_path, 'rb') as f:
                        return self.send_bytes(f.read(), 'text/html; charset=utf-8')
                if path == '/api/scenes':
                    return self.send_json({'origin': db['origin'], 'scenes': db['scenes']})
                if path == '/api/labels':
                    with labels_lock:
                        return self.send_json(read_json(labels_path, {'version': 1, 'labels': {}}))
                if path == '/api/pairs':
                    with labels_lock:
                        return self.send_json(read_json(pairs_path, {'version': 1, 'pairs': {}}))
                if path == '/api/selection':
                    return self.send_json(read_json(result_path, {'missing': True}))
                if path == '/api/apply/plan':
                    try:
                        return self.send_json(apply_plan())
                    except ValueError as e:
                        return self.send_json({'error': str(e)})
                if path == '/api/decisions':
                    with labels_lock:
                        return self.send_json(read_json(decisions_path, {'version': 1, 'decisions': {}}))
                if path == '/api/risk':
                    return self.send_json(read_json(os.path.join(sel_dir, 'risk.json'), {'missing': True}))
                if path.startswith('/api/frames/'):
                    # all frames of one channel in a scene: /api/frames/scene-0026?ch=CAM_FRONT
                    ch = (parse_qs(urlparse(self.path).query).get('ch') or ['CAM_FRONT'])[0]
                    lst = db['frames'].get(path.rsplit('/', 1)[-1], {}).get(ch)
                    if lst is None:
                        return self.send_error(404, 'unknown scene or channel')
                    return self.send_json({'ts': [t for t, _ in lst], 'fn': [f for _, f in lst]})
                if path.startswith('/api/sample/'):
                    payload = sample_payload(db, path.rsplit('/', 1)[-1])
                    if payload is None:
                        return self.send_error(404, 'unknown sample')
                    return self.send_json(payload)
                if path.startswith('/lidar/') and np is not None:
                    # compact LiDAR for display: x, y, z as int16 centimetres (6 B/point instead of 20 B float32 x 5)
                    full = os.path.realpath(os.path.join(root, path[len('/lidar/'):]))
                    if not full.startswith(root + os.sep) or not os.path.isfile(full):
                        return self.send_error(404)
                    xyz = np.fromfile(full, dtype=np.float32).reshape(-1, 5)[:, :3]
                    body = np.clip(np.round(xyz * 100), -32767, 32767).astype('<i2').tobytes()
                    return self.send_bytes(body, 'application/octet-stream', cache=True)
                if path.startswith('/data/'):
                    full = os.path.realpath(os.path.join(root, path[len('/data/'):]))
                    if not full.startswith(root + os.sep) or not os.path.isfile(full):
                        return self.send_error(404)
                    with open(full, 'rb') as f:
                        body = f.read()
                    ctype = mimetypes.guess_type(full)[0] or 'application/octet-stream'
                    return self.send_bytes(body, ctype, cache=True)
                return self.send_error(404)
            except ConnectionError:              # client went away (Broken pipe / reset / aborted, WinError 10053)
                pass

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            handler = {'/api/labels': update_label, '/api/pairs': update_pair, '/api/decisions': update_decision,
                       '/api/apply': run_apply}.get(path)
            if handler is None:
                return self.send_error(404)
            try:
                n = int(self.headers.get('Content-Length', 0))
                if n <= 0 or n > 65536:
                    return self.send_error(400, 'bad body size')
                return self.send_json(handler(json.loads(self.rfile.read(n))))
            except (ValueError, json.JSONDecodeError) as e:
                return self.send_error(400, str(e))
            except ConnectionError:              # client went away (Broken pipe / reset / aborted, WinError 10053)
                pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.dirname(HERE))
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    db = load_db(args.dataroot, args.version)

    class Server(ThreadingHTTPServer):
        request_queue_size = 128          # default 5: bursts of parallel image/LiDAR requests got "connection refused"
        daemon_threads = True
    server = Server(('127.0.0.1', args.port), make_handler(db, args.dataroot, args.version))
    url = f'http://localhost:{args.port}'
    print(f'[viewer] serving {args.dataroot} at {url}  (Ctrl+C to stop)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.exit(main())
