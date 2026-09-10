#!/usr/bin/env python3
"""Live crawl dashboard poller. Writes a static snapshot HTML every 60s.

KEY stays here (server-side); the browser only reads the snapshot file.
Also reusable for the 10M build: --build, --repo, --run, --workers.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ''


def qdrant_total(env, collection):
    try:
        base = env['QDRANT_URL'].rstrip('/') + f'/collections/{collection}'
        req = urllib.request.Request(base, headers={'api-key': env['QDRANT_API_KEY']})
        return json.load(urllib.request.urlopen(req, timeout=30))['result']['points_count']
    except Exception:
        return None


def qdrant_count(env, collection, build):
    try:
        base = env['QDRANT_URL'].rstrip('/') + f'/collections/{collection}'
        body = {'filter': {'must': [{'key': 'build_id', 'match': {'value': build}}]}}
        req = urllib.request.Request(base + '/points/count', data=json.dumps(body).encode(),
                                     headers={'api-key': env['QDRANT_API_KEY'],
                                              'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=30))['result']['count']
    except Exception:
        return None


def disk_str(load):
    try:
        avail_gb = int(load.strip().split('\n')[1]) / 1e9
        return f'{avail_gb:.0f}G free'
    except Exception:
        return '?'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', default='qfv-2')
    ap.add_argument('--repo', default='souiiii/imgsearch')
    ap.add_argument('--run', default='34486584760')
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--out', default=str(Path.home() / 'Downloads/imgsearch-crawl.html'))
    ap.add_argument('--hist', default='/tmp/qfv-history.jsonl')
    args = ap.parse_args()

    env = dict(l.strip().split('=', 1) for l in open(REPO_ROOT / '.env')
               if '=' in l and not l.startswith('#'))
    started = time.time()
    created = None
    while True:
        now = time.time()
        try:
            jobs = json.loads(sh(['gh', 'run', 'view', args.run, '--repo', args.repo,
                                          '--json', 'jobs', 'createdAt']) or '{}')
            created = jobs.get('createdAt', created)
        except Exception:
            pass
        st = sh(['gh', 'run', 'view', args.run, '--repo', args.repo,
                         '--json', 'jobs',
                         '--jq', '[.jobs[] | .status] | group_by(.) | map({(.[0]): length}) | add'])
        count = qdrant_count(env, 'images-int8-disk', args.build)
        total = qdrant_total(env, 'images-int8-disk')
        load = sh(['ssh', '-i', str(Path.home() / '.ssh/mac-app'), '-o', 'BatchMode=yes',
                           '-o', 'ConnectTimeout=8', 'ubuntu@89.167.20.186',
                           'awk \'{print $1" (1m) · "$2" (5m)"}\' /proc/loadavg; '
                           'df -B1 --output=avail / | tail -1'])
        snap = {'t': now, 'count': count}
        try:
            with open(args.hist, 'a') as f:
                f.write(json.dumps(snap) + '\n')
            hist = [json.loads(l) for l in open(args.hist) if l.strip()]
        except Exception:
            hist = [snap]
        window = [h for h in hist if now - h['t'] <= 360 and h['count'] is not None]
        pace5 = ((window[-1]['count'] - window[0]['count']) / max(now - window[0]['t'], 1) * 60
                 if len(window) > 1 else None)
        first = next((h for h in hist if h['count'] is not None), None)
        pace_all = ((count - first['count']) / max(now - first['t'], 1) * 60
                    if first and count is not None and now > first['t'] else None)
        # Yield unknown until categories exhaust; 300k indexed is the
        # mid-estimate projection (owner-approved, not a target).
        eta_300k = (300000 - count) / pace5 / 60 if pace5 and pace5 > 0 and count else None
        el = ''
        if created:
            try:
                t0 = datetime.datetime.fromisoformat(created.replace('Z', '+00:00')).timestamp()
                el = f'{int((now - t0) // 60)} min (window ends ~{int(270 - (now - t0) // 60)} min)'
            except Exception:
                pass
        gen = datetime.datetime.now().strftime('%H:%M:%S')
        rows = [
            ('Indexed (build %s)' % args.build, f'{count:,}' if count is not None else '?'),
            ('Total in storage (all builds)', f'{total:,}' if total is not None else '?'),
            ('Pace (trailing 5 min)', f'{pace5:,.0f}/min' if pace5 else '?'),
            ('Pace (overall)', f'{pace_all:,.0f}/min' if pace_all else '?'),
            ('Elapsed (4.5h window)', el or '?'),
            ('Runner jobs', st.strip() or '?'),
            ('Box load (4 vCPU box)', load.strip().split('\n')[0] if load.strip() else '?'),
            ('Disk free (75G box)', disk_str(load)),
            ('ETA to ~300k (at 5-min pace)', f'~{eta_300k:.1f}h' if eta_300k else '?'),
        ]
        trs = '\n'.join(f'<tr><td>{k}</td><td><b>{v}</b></td></tr>' for k, v in rows)
        html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=60>
<title>Q/F/V crawl</title>
<style>body{{background:#14161a;color:#e8e9eb;font:16px/1.5 system-ui;max-width:640px;margin:32px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}td{{padding:8px;border-bottom:1px solid #333}}td:last-child{{text-align:right}}small{{color:#8a93a3}}</style>
</head><body><h1>Q/F/V crawl <small>build {args.build} · {args.workers} lanes</small></h1>
<table>{trs}</table>
<p><small>Snapshot {gen} · refreshes every 60s · ~300k indexed mid-estimate · ETA uses trailing 5-min pace · goal: full 487k gross layer</small></p>
</body></html>"""
        Path(args.out).write_text(html)
        time.sleep(60)


if __name__ == '__main__':
    sys.exit(main())
