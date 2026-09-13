#!/usr/bin/env python3
"""Seed the special-collection builds: NASA, museum masterpieces, medical.

Each source becomes small queue units, exactly like the celeb build:

    nasa      one unit per (query, page) of images-api.nasa.gov search
    met       one unit per 100 object IDs (CC0 + image, highlights first)
    wellcome  one unit per (query, page) of the Wellcome images API

The query lists below are curated for recognisability ("famous ones"): the
NASA list is missions and science icons, the Met list is famous artists and
civilisations, Wellcome is the canonical medical/biology subjects.

Usage:
    python special_targets.py --source nasa --seed \
        --queue-url https://api.framedrop.website/crawl-queue \
        --app-key <key> --build nasa-v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

NASA_API = 'https://images-api.nasa.gov'
MET_API = 'https://collectionapi.metmuseum.org/public/collection/v1'
WELLCOME_API = 'https://api.wellcomecollection.org/catalogue/v2'
UA = {'User-Agent': 'FrameDrop/1.0 (special collections; fent.zahid@gmail.com)'}

NASA_QUERIES = [
    'apollo 11', 'apollo program', 'artemis', 'saturn v', 'space shuttle',
    'international space station', 'hubble', 'james webb', 'earthrise',
    'blue marble', 'moon landing', 'neil armstrong', 'buzz aldrin', 'michael collins',
    'mars rover', 'perseverance', 'curiosity', 'ingenuity', 'voyager', 'cassini',
    'juno jupiter', 'new horizons pluto', 'solar system', 'nebula', 'galaxy',
    'black hole', 'milky way', 'andromeda', 'supernova', 'star cluster',
    'earth from space', 'earth observation', 'hurricane from space', 'aurora from space',
    'sun', 'solar flare', 'eclipse', 'comet', 'asteroid', 'meteor',
    'astronaut', 'spacewalk', 'spacesuit', 'launch', 'rocket', 'satellite',
    'deep space network', 'kennedy space center', 'mission control', 'apollo 13',
    'gemini program', 'mercury program', 'x-15', 'sr-71', 'f-18', 'aircraft',
    'nasa science', 'earth science', 'climate', 'ocean from space', 'ice sheet',
    'grand canyon from space', 'city from space', 'night lights from space',
    'volcano from space', 'wildfire from space', 'solar sail', 'ion engine',
]

MET_QUERIES = [
    # Art-historical heavyweights
    'van gogh', 'monet', 'renoir', 'degas', 'cezanne', 'manet', 'gauguin',
    'rembrandt', 'vermeer', 'rubens', 'bruegel', 'bosch', 'durer', 'hokusai',
    'hiroshige', 'utamaro', 'picasso', 'matisse', 'modigliani', 'rodin',
    'turner', 'constable', 'goya', 'velazquez', 'el greco', 'caravaggio',
    'raphael', 'michelangelo', 'leonardo', 'botticelli', 'titian', 'veronese',
    'klimt', 'munch', 'vinci', 'ingres', 'delacroix', 'courbet', 'corot',
    'sargent', 'whistler', 'homer', 'hopper', 'okeeffe', 'pollock', 'warhol',
    'lichtenstein', 'rothko', 'klee', 'kandinsky', 'mondrian', 'miro',
    # Collections and civilisations
    'egyptian', 'greek', 'roman', 'byzantine', 'islamic', 'chinese', 'japanese',
    'korean', 'indian', 'southeast asian', 'african', 'oceanic', 'precolumbian',
    'ancient near eastern', 'medieval', 'renaissance', 'baroque', 'impressionism',
    'american wing', 'arms and armor', 'samurai', 'ancient glass', 'jewelry',
    'gold', 'silver', 'bronze', 'marble sculpture', 'terracotta', 'ceramics',
    'porcelain', 'furniture', 'costume', 'textile', 'tapestry', 'manuscript',
    'illuminated manuscript', 'photographs', 'drawings', 'prints', 'portrait',
    'landscape', 'still life', 'mythology', 'religious art', 'cathedral',
]

WELLCOME_QUERIES = [
    'anatomy', 'human body', 'skeleton', 'skull', 'heart', 'brain', 'lungs',
    'liver', 'kidney', 'stomach', 'muscles', 'nerves', 'eye', 'ear', 'teeth',
    'hands', 'feet', 'blood', 'blood cells', 'dna', 'chromosome', 'gene',
    'cell', 'cells microscope', 'bacteria', 'virus', 'influenza', 'coronavirus',
    'vaccination', 'vaccine', 'immunity', 'antibiotic', 'penicillin', 'surgery',
    'surgeon', 'operating theatre', 'hospital', 'nurse', 'doctor', 'midwife',
    'childbirth', 'child health', 'plague', 'cholera', 'smallpox', 'tuberculosis',
    'malaria', 'yellow fever', 'syphilis', 'leprosy', 'cancer', 'tumour',
    'pharmacy', 'apothecary', 'medicine bottle', 'pill', 'herbal medicine',
    'botany', 'plants medicinal', 'microscope', 'laboratory', 'chemistry',
    'x-ray', 'radiography', 'scan', 'prosthetics', 'dentistry', 'psychiatry',
    'asylum', 'ambulance', 'first aid', 'public health', 'sanitation', 'water supply',
]


def nasa_units(client):
    units = []
    for query in NASA_QUERIES:
        try:
            r = client.get(f'{NASA_API}/search',
                           params={'q': query, 'media_type': 'image', 'page_size': 1})
            r.raise_for_status()
            total = int(r.json()['collection']['metadata'].get('total_hits') or 0)
        except Exception as exc:
            print(f'  nasa query failed {query!r}: {type(exc).__name__}', file=sys.stderr)
            continue
        pages = min(30, (total + 99) // 100)
        for page in range(1, pages + 1):
            units.append({'source': 'nasa', 'query': query, 'page': page,
                          'max_pages': pages})
        time.sleep(0.2)
    return units


def met_units(client, ids_file=''):
    if ids_file:
        ids = json.load(open(ids_file))
        print(f'  met: {len(ids)} ids from {ids_file}', file=sys.stderr)
        return [{'source': 'met', 'ids': ids[i:i + 100]}
                for i in range(0, len(ids), 100)]
    ids, seen = [], set()
    for query in MET_QUERIES:
        for highlight in (True, False):
            params = {'isPublicDomain': 'true', 'hasImages': 'true', 'q': query}
            if highlight:
                params['isHighlight'] = 'true'
            try:
                r = client.get(f'{MET_API}/search', params=params)
                r.raise_for_status()
            except Exception as exc:
                print(f'  met query failed {query!r}: {type(exc).__name__}',
                      file=sys.stderr)
                continue
            for object_id in (r.json().get('objectIDs') or [])[:2000]:
                if object_id not in seen:
                    seen.add(object_id)
                    ids.append(object_id)
            time.sleep(0.2)
    print(f'  met: {len(ids)} unique objects', file=sys.stderr)
    return [{'source': 'met', 'ids': ids[i:i + 100]}
            for i in range(0, len(ids), 100)]


def wellcome_units(client):
    units = []
    for query in WELLCOME_QUERIES:
        try:
            r = client.get(f'{WELLCOME_API}/images',
                           params={'query': query, 'pageSize': 1})
            r.raise_for_status()
            total = int(r.json().get('totalResults') or 0)
        except Exception as exc:
            print(f'  wellcome query failed {query!r}: {type(exc).__name__}',
                  file=sys.stderr)
            continue
        pages = min(50, (total + 99) // 100)
        for page in range(1, pages + 1):
            units.append({'source': 'wellcome', 'query': query, 'page': page,
                          'max_pages': pages})
        time.sleep(0.2)
    return units


def seed(units, queue_url, app_key, build, batch=400):
    headers = {'X-FrameDrop-Key': app_key} if app_key else {}
    inserted = 0
    pending = []

    def flush():
        nonlocal inserted, pending
        if not pending:
            return
        r = client.post(queue_url.rstrip('/') + '/seed',
                        json={'build': build, 'units': pending})
        r.raise_for_status()
        inserted += r.json().get('inserted', 0)
        pending = []

    with httpx.Client(timeout=60, headers={**UA, **headers}) as client:
        for unit in units:
            pending.append(unit)
            if len(pending) >= batch:
                flush()
        flush()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True,
                        choices=['nasa', 'met', 'wellcome'])
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--queue-url',
                        default='https://api.framedrop.website/crawl-queue')
    parser.add_argument('--app-key', default='')
    parser.add_argument('--met-ids', default='')
    parser.add_argument('--build', default='')
    args = parser.parse_args()

    with httpx.Client(timeout=60, headers=UA) as client:
        print(f'building {args.source} units…', file=sys.stderr)
        if args.source == 'nasa':
            units = nasa_units(client)
        elif args.source == 'met':
            units = met_units(client, args.met_ids)
        else:
            units = wellcome_units(client)
    print(f'{len(units)} units', file=sys.stderr)

    if args.seed:
        build = args.build or f'{args.source}-v1'
        inserted = seed(units, args.queue_url, args.app_key, build)
        print(f'seeded {inserted} new units into {build} '
              f'({len(units)} posted, duplicates ignored)', file=sys.stderr)


if __name__ == '__main__':
    main()
