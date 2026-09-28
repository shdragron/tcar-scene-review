"""Apply the reviewed selection to the dataset: take the deleted scenes out and renumber the rest without gaps.

The list comes from selection/final_selection.json (the rule's proposals overridden by the viewer's human
decisions; written by the viewer and by curate.py).

    python tools/apply_selection.py                        # plan only (nothing changes)
    python tools/apply_selection.py --apply                # do it
    python tools/apply_selection.py --undo _removed/<stamp>

Nothing is deleted. Everything taken out goes to <dataroot>/_removed/<stamp>/ and manifest.json there lists
every move, so --undo puts the dataset back exactly:
  1. backup   the tables (v1.0-trainval/*.json), *.import.json and selection/ are copied first
  2. tables   deleted scenes, their samples, sample_data (key frames and sweeps), ego poses and annotations
              are removed; a log / map left without scenes is dropped
  3. files    the deleted scenes' samples/ + sweeps/ files and can_bus/ files are moved into the backup
  4. names    the remaining scenes take the dataset's sorted names in order, so there is no gap: with
              2, 3, 4 and 3 deleted, 4 becomes 3. The names are the first N of the official nuScenes train
              split (what the converter assigned), so the devkit still sees them all as train;
              can_bus files and *.import.json follow the new names
  5. caches   selection/ files keyed by scene name are re-keyed (objects, stop events, risk, LiDAR, CLIP), and so
              are the human decisions / labels of the scenes that stay; the old result and final list stay in
              the backup (the viewer reruns the rule on the new dataset)
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = 'v1.0-trainval'


def _read(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write(path, obj, indent=None):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def plan(dataroot):
    """What --apply would do, without changing anything."""
    final = _read(os.path.join(dataroot, 'selection', 'final_selection.json'))
    if not final:
        raise ValueError('selection/final_selection.json 없음 -> python tools/curate.py 먼저')
    tdir = os.path.join(dataroot, VERSION)
    scenes = _read(os.path.join(tdir, 'scene.json'))
    names = sorted(s['name'] for s in scenes)
    deleted = sorted(n for n in final['deleted'] if n in names)
    missing = sorted(set(final['deleted']) - set(names))
    remaining = [n for n in names if n not in deleted]
    rename = {old: new for old, new in zip(remaining, names[:len(remaining)]) if old != new}
    sizes = _read(os.path.join(dataroot, 'selection', 'scene_bytes.json'), {}).get('scenes', {})
    tok = {s['name']: s['token'] for s in scenes}
    nbytes = sum((sizes.get(tok[n]) or {}).get('bytes', 0) for n in deleted)
    return {'deleted': deleted, 'missing': missing, 'rename': rename, 'n_before': len(names), 'n_after': len(remaining),
            'bytes_moved': nbytes, 'not_reviewed': final['counts'].get('proposals_not_reviewed', 0),
            'human_decisions': final['counts'].get('human_decisions', 0)}


def _rekey(d, rename, deleted):
    """dict keyed by scene name -> deleted scenes dropped, the others under their new names."""
    return {rename.get(k, k): v for k, v in d.items() if k not in deleted}


def apply(dataroot, log=print):
    p = plan(dataroot)
    if p['missing']:
        raise ValueError(f"목록에 있지만 데이터에 없는 scene: {p['missing']}")
    deleted, rename = set(p['deleted']), p['rename']
    if not deleted:
        return dict(p, backup=None)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    B = os.path.join(dataroot, '_removed', stamp)
    tdir, sel = os.path.join(dataroot, VERSION), os.path.join(dataroot, 'selection')
    os.makedirs(B)
    man = {'stamp': stamp, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'deleted': sorted(deleted), 'rename': rename,
           'moves': [], 'done': False}

    def save_manifest():
        _write(os.path.join(B, 'manifest.json'), man, indent=1)

    def move(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        man['moves'].append([src, dst])

    # 1. backup (copies) --------------------------------------------------------------------------------
    shutil.copytree(tdir, os.path.join(B, 'tables'))
    os.makedirs(os.path.join(B, 'import'))
    for f in glob.glob(os.path.join(dataroot, '*.import.json')):
        shutil.copy2(f, os.path.join(B, 'import'))
    if os.path.isdir(sel):
        shutil.copytree(sel, os.path.join(B, 'selection'))
    save_manifest()
    log(f'[apply] backup -> {B}')

    # 2. tables -----------------------------------------------------------------------------------------
    T = {os.path.basename(f)[:-5]: _read(f) for f in glob.glob(os.path.join(tdir, '*.json'))}
    del_scene = {s['token'] for s in T['scene'] if s['name'] in deleted}
    del_sample = {s['token'] for s in T['sample'] if s['scene_token'] in del_scene}
    del_sd = [x for x in T['sample_data'] if x['sample_token'] in del_sample]
    del_ego = {x['ego_pose_token'] for x in del_sd}
    del_sd_tok = {x['token'] for x in del_sd}
    T['scene'] = [dict(s, name=rename.get(s['name'], s['name'])) for s in T['scene'] if s['token'] not in del_scene]
    T['sample'] = [s for s in T['sample'] if s['token'] not in del_sample]
    T['sample_data'] = [x for x in T['sample_data'] if x['token'] not in del_sd_tok]
    T['ego_pose'] = [e for e in T['ego_pose'] if e['token'] not in del_ego]
    if T.get('sample_annotation'):
        T['sample_annotation'] = [a for a in T['sample_annotation'] if a['sample_token'] not in del_sample]
        ann = {a['token'] for a in T['sample_annotation']}
        T['instance'] = [i for i in T.get('instance', []) if i.get('first_annotation_token') in ann]
    live_logs = {s['log_token'] for s in T['scene']}
    T['log'] = [l for l in T['log'] if l['token'] in live_logs]
    for m in T.get('map', []):
        m['log_tokens'] = [t for t in m['log_tokens'] if t in live_logs]
    for k, v in T.items():
        _write(os.path.join(tdir, k + '.json'), v, indent=2)          # same layout as the converter wrote
    log(f"[apply] tables: -{len(del_scene)} scenes, -{len(del_sample)} samples, -{len(del_sd)} sample_data")

    # 3. files ------------------------------------------------------------------------------------------
    for x in del_sd:
        src = os.path.join(dataroot, x['filename'])
        if os.path.isfile(src):
            move(src, os.path.join(B, 'files', x['filename']))
    for n in sorted(deleted):
        for f in glob.glob(os.path.join(dataroot, 'can_bus', n + '_*')):
            move(f, os.path.join(B, 'can_bus', os.path.basename(f)))
    save_manifest()
    log(f"[apply] moved {len(man['moves'])} files into the backup")

    # 4. names: can_bus files in two steps (a new name can be an old name still in use), import.json -------
    cb = os.path.join(dataroot, 'can_bus')
    staged = []
    for old, new in rename.items():
        for f in glob.glob(os.path.join(cb, old + '_*')):
            tmp = os.path.join(cb, '__renaming__' + new + os.path.basename(f)[len(old):])
            move(f, tmp)
            staged.append(tmp)
    for tmp in staged:
        move(tmp, os.path.join(cb, os.path.basename(tmp)[len('__renaming__'):]))
    for f in glob.glob(os.path.join(dataroot, '*.import.json')):
        j = _read(f)
        j['scenes'] = [dict(s, name=rename.get(s['name'], s['name'])) for s in j.get('scenes', []) if s['name'] not in deleted]
        _write(f, j, indent=1)
    save_manifest()
    log(f'[apply] renamed {len(rename)} scenes')

    # 5. selection/ -------------------------------------------------------------------------------------
    for name in ('objects.json', 'stop_events.json', 'risk.json'):
        path = os.path.join(sel, name)
        j = _read(path)
        if j and isinstance(j.get('scenes'), dict):
            j['scenes'] = _rekey(j['scenes'], rename, deleted)
            _write(path, j, indent=1)
    for name in ('lidar_cache.pkl', 'risk_cache.pkl'):
        path = os.path.join(sel, name)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                c = pickle.load(f)
            with open(path, 'wb') as f:
                pickle.dump(_rekey(c, rename, deleted), f)
    for path in glob.glob(os.path.join(sel, 'image_emb_*.npz')):
        import numpy as np
        z = np.load(path)
        keep = [i for i, n in enumerate(z['names']) if str(n) not in deleted]
        np.savez(path, names=np.array([rename.get(str(z['names'][i]), str(z['names'][i])) for i in keep]), emb=z['emb'][keep])
    for name, key in (('human_labels.json', 'labels'), ('pair_labels.json', 'pairs'), ('human_decisions.json', 'decisions')):
        # decisions on the scenes that stay (e.g. 남기기 against the rule) follow them to their new names, so the
        # rule rerun on the new data cannot silently propose them again
        path = os.path.join(sel, name)
        j = _read(path)
        if j and isinstance(j.get(key), dict):
            if key == 'pairs':
                j[key] = {'|'.join(sorted(rename.get(a, a) for a in k.split('|'))): v for k, v in j[key].items()
                          if not set(k.split('|')) & deleted}
            else:
                j[key] = _rekey(j[key], rename, deleted)
            _write(path, j, indent=1)
    for name in ('lidar_diff.json', 'scene_bytes.json'):          # keyed by token; drop by token, then rename
        path = os.path.join(sel, name)                            # (a new name can equal a deleted scene's old name)
        j = _read(path)
        if j and isinstance(j.get('scenes'), dict):
            j['scenes'] = {t: v for t, v in j['scenes'].items() if t not in del_scene}
            for v in j['scenes'].values():
                if isinstance(v, dict) and v.get('name') in rename:
                    v['name'] = rename[v['name']]
            _write(path, j, indent=1)
    for name in ('result.json', 'final_selection.json', 'pairs.csv', 'scene_features.csv'):
        path = os.path.join(sel, name)                            # old names; the copies stay in the backup
        if os.path.isfile(path):
            os.remove(path)
    with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  === 삭제 확정 적용: {len(deleted)}개 scene을 {B} 로 이동, "
                f"{len(rename)}개 이름 당김 ({', '.join(f'{a[-4:]}->{b[-4:]}' for a, b in rename.items())}) ===\n")
    man['done'] = True
    save_manifest()
    log('[apply] done')
    return dict(p, backup=B)


def undo(backup, dataroot, log=print):
    """Put the dataset back as it was before the --apply that wrote `backup`."""
    man = _read(os.path.join(backup, 'manifest.json'))
    if not man:
        raise ValueError(f'{backup}/manifest.json 없음')
    for src, dst in reversed(man['moves']):
        if os.path.exists(dst):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            os.replace(dst, src)
    tdir = os.path.join(dataroot, VERSION)
    for f in glob.glob(os.path.join(backup, 'tables', '*.json')):
        shutil.copy2(f, os.path.join(tdir, os.path.basename(f)))
    for f in glob.glob(os.path.join(backup, 'import', '*.json')):
        shutil.copy2(f, os.path.join(dataroot, os.path.basename(f)))
    sel = os.path.join(dataroot, 'selection')
    for f in glob.glob(os.path.join(backup, 'selection', '*')):
        shutil.copy2(f, os.path.join(sel, os.path.basename(f)))
    with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  === 되돌림: {backup} ===\n")
    open(os.path.join(backup, 'UNDONE'), 'w').close()
    log(f"[undo] restored {len(man['moves'])} moved files, tables, import.json and selection/ from {backup}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.dirname(HERE))
    ap.add_argument('--apply', action='store_true', help='move the deleted scenes out and renumber (default: plan only)')
    ap.add_argument('--undo', metavar='BACKUP_DIR', help='restore from a backup written by --apply')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass
    if args.undo:
        return undo(args.undo, args.dataroot)
    p = plan(args.dataroot)
    print(f"scene {p['n_before']} -> {p['n_after']}  (삭제 {len(p['deleted'])}: {', '.join(n[-4:] for n in p['deleted'])}, "
          f"{p['bytes_moved'] / 1e9:.1f} GB 이동)")
    print('이름 당김: ' + (', '.join(f'{a[-4:]}->{b[-4:]}' for a, b in p['rename'].items()) or '없음'))
    if p['not_reviewed']:
        print(f"주의: 사람이 아직 보지 않은 규칙 제안 {p['not_reviewed']}개가 포함됨")
    if args.apply:
        r = apply(args.dataroot)
        print(f"완료. 되돌리기: python tools/apply_selection.py --undo \"{r['backup']}\"")
    else:
        print('(계획만 표시 · 실제 적용은 --apply)')


if __name__ == '__main__':
    sys.exit(main())
