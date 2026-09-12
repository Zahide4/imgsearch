#!/usr/bin/env python3
"""Celebrity target list and queue seeding for the modern-figures build.

Fame is proxied by Wikidata sitelinks: the number of language editions that
carry an article about the person. The most globally recognisable people sit
at 60-200+ sitelinks; ranking by it produces an "A-list" without anyone
hand-curating names.

Only people with a Commons image and a mainstream occupation (film, music,
sport, presenting, modelling, politics) enter the list. The result feeds the
crawl queue as small units:

    one Openverse unit per person
    + one Commons unit per (clean licence, width band) = 16 more

Commons units reuse the topic-search path (licence and width shards); the
Openverse unit pages the CC0/PDM/CC-BY results. Both stop on their own when
a person's clean pool is exhausted.

Usage:
    python celeb_targets.py --top 3000 --out celebrities.json
    python celeb_targets.py --out celebrities.json \
        --seed --queue-url https://api.framedrop.website/crawl-queue \
        --app-key <key> --build celeb-v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

WDQS = 'https://query.wikidata.org/sparql'
UA = {'User-Agent': 'FrameDrop/1.0 (celeb targets; fent.zahid@gmail.com)',
      'Accept': 'application/sparql-results+json'}

# Mainstream occupations. Anything a magazine cover would call a "star".
OCCUPATIONS = [
    'Q33999',     # actor
    'Q10800557',  # film actor
    'Q10798782',  # television actor
    'Q177220',    # singer
    'Q2252262',   # rapper
    'Q639669',    # musician
    'Q245068',    # comedian
    'Q2526255',   # film director
    'Q4610556',   # model
    'Q2405480',   # television presenter
    'Q937857',    # association football player
    'Q3665646',   # basketball player
    'Q10833314',  # tennis player
    'Q12299841',  # cricketer
    'Q11338576',  # boxer
    'Q378622',    # racing driver
    'Q2066131',   # athlete
    'Q82955',     # politician
    'Q17125263',  # YouTuber
    'Q183945',    # record producer
]

CLEAN_LICENCES = ['CC-Zero', 'CC-BY-4.0', 'CC-BY-2.0', 'CC-BY-3.0',
                  'CC-PD-Mark', 'PD-old-100-expired', 'PD-self', 'PD-1996']
WIDTH_BANDS = ['filew:>3000', 'filew:<2999']


def sparql(top: int, min_sitelinks: int, living_only: bool) -> list[dict]:
    """The fame-ranked list, retrying with a lower bar while WDQS is busy."""
    threshold = max(min_sitelinks, 10)
    while threshold >= 5:
        values = ' '.join(f'wd:{q}' for q in OCCUPATIONS)
        living = 'FILTER NOT EXISTS { ?person wdt:P570 ?death . }' if living_only else ''
        query = f"""
        SELECT ?person ?personLabel ?sitelinks WHERE {{
          ?person wdt:P31 wd:Q5; wdt:P18 ?image;
                  wikibase:sitelinks ?sitelinks;
                  wdt:P106 ?occupation .
          VALUES ?occupation {{ {values} }}
          FILTER(?sitelinks >= {threshold})
          {living}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?sitelinks)
        LIMIT {top}
        """
        for attempt in range(3):
            try:
                r = httpx.get(WDQS, params={'query': query}, headers=UA,
                              timeout=180)
                if r.status_code == 200:
                    rows = r.json()['results']['bindings']
                    out = [{'qid': row['person']['value'].rsplit('/', 1)[-1],
                            'name': row['personLabel']['value'],
                            'sitelinks': int(row['sitelinks']['value'])}
                           for row in rows]
                    print(f'sitelinks >= {threshold}: {len(out)} people',
                          file=sys.stderr)
                    return out
                print(f'WDQS {r.status_code} (threshold {threshold})',
                      file=sys.stderr)
            except Exception as exc:
                print(f'WDQS {type(exc).__name__} (threshold {threshold})',
                      file=sys.stderr)
            time.sleep(15 * (attempt + 1))
        threshold //= 2
    raise SystemExit('WDQS unavailable; pass --names-file to skip it')


def units_for(name: str) -> list[dict]:
    units = [{'name': name, 'source': 'openverse'}]
    for licence in CLEAN_LICENCES:
        for band in WIDTH_BANDS:
            units.append({'name': name, 'source': 'commons',
                          'lic': licence, 'band': band})
    return units


def seed(people: list[dict], queue_url: str, app_key: str, build: str,
         batch: int = 400) -> int:
    headers = {'X-FrameDrop-Key': app_key} if app_key else {}
    url = queue_url.rstrip('/') + '/seed'
    inserted = 0
    pending: list[dict] = []
    with httpx.Client(timeout=60, headers=headers) as client:
        def flush():
            nonlocal inserted, pending
            if not pending:
                return
            r = client.post(url, json={'build': build, 'units': pending})
            r.raise_for_status()
            inserted += r.json().get('inserted', 0)
            pending = []
        for person in people:
            pending.extend(units_for(person['name']))
            if len(pending) >= batch:
                flush()
        flush()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=3000)
    parser.add_argument('--min-sitelinks', type=int, default=15)
    parser.add_argument('--living-only', action='store_true')
    parser.add_argument('--names-file', default='',
                        help='skip Wikidata and read a JSON list of names or '
                             '{name} objects (a fallback when WDQS is down)')
    parser.add_argument('--out', default='celebrities.json')
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--queue-url', default='https://api.framedrop.website/crawl-queue')
    parser.add_argument('--app-key', default='')
    parser.add_argument('--build', default='celeb-v1')
    args = parser.parse_args()

    if args.names_file:
        raw = json.load(open(args.names_file))
        people = [item if isinstance(item, dict) else {'name': item}
                  for item in raw]
    else:
        people = sparql(args.top, args.min_sitelinks, args.living_only)
    with open(args.out, 'w') as handle:
        json.dump(people, handle, indent=1, ensure_ascii=False)
    print(f'{len(people)} people -> {args.out}', file=sys.stderr)

    if args.seed:
        inserted = seed(people, args.queue_url, args.app_key, args.build)
        print(f'seeded {inserted} new units into {args.build} '
              f'({len(people) * 17} posted, duplicates ignored)', file=sys.stderr)


if __name__ == '__main__':
    main()
