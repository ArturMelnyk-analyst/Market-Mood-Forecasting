"""Safely merge the six-session Arena evidence backfill into the local research CSV.

Usage from Market-Mood-Forecasting project root:
  python merge_arena_peer_backfill.py --backfill data/processed/es_arena_research_v0_1/es_peer_backfill_20261020_23.csv
  python merge_arena_peer_backfill.py --backfill data/processed/es_arena_research_v0_1/es_peer_backfill_20261020_23.csv --apply

No network access. Without --apply, nothing is modified. With --apply, both
input CSVs are backed up before atomic replacement.

Evidence caveat: model probabilities are rounded values rendered on Arena's public
pages. counts_for_score=true is based on the displayed scored prediction cards,
not on independent agent-specific API receipts. See source audit CSV.
"""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import os
import shutil
import tempfile

RESEARCH = Path('data/processed/es_arena_research_v0_1')
PEER_COLUMNS = ['challenge_id','agent_id','agent_name','p_bearish','p_neutral','p_bullish','submitted_at_utc','counts_for_score','source_url']
EXPECTED_DEADLINES = {
 '118cefc4-96a8-4e47-b617-73af53c67d03': ('2026-10-20','2026-10-20T14:00:00+00:00'),
 '0a10e1af-cc9d-47f4-a6aa-ef5e4346bcd0': ('2026-10-21','2026-10-21T14:00:00+00:00'),
 'f3f54d5f-275b-4241-b52a-026764fefd64': ('2026-10-22','2026-10-22T14:00:00+00:00'),
 '2d0b3370-29af-4483-94b6-c1b553d8a3c6': ('2026-10-23','2026-10-23T14:00:00+00:00'),
}
OWN_AGENT_ID = 'agt_f7b9b0827875'


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing file: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'Missing CSV header: {path}')
        columns=reader.fieldnames
        rows=list(reader)
    if any(None in row for row in rows):
        raise ValueError(f'Malformed CSV row/extra separator: {path}')
    return columns, rows


def as_utc(value):
    value=value.strip()
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:
        raise ValueError(f'Timestamp lacks timezone: {value}')
    return dt.astimezone(timezone.utc)


def write_atomic(path, columns, rows):
    # Write in the same directory so os.replace remains an atomic single-file swap.
    name=None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                             dir=path.parent, prefix='.peer_merge_',
                                             suffix='.csv', delete=False) as f:
            name=f.name
            writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\r\n')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name,path)
    finally:
        if name and os.path.exists(name):os.unlink(name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backfill',required=True,type=Path)
    parser.add_argument('--apply',action='store_true',help='Commit validated changes after backups (default: dry-run)')
    args=parser.parse_args()
    peer_file=RESEARCH/'arena_peer_forecasts.csv'
    official_file=RESEARCH/'arena_official_settlements.csv'
    source_cols,source_rows=read_csv(args.backfill)
    peer_cols,peer_rows=read_csv(peer_file)
    official_cols,official_rows=read_csv(official_file)
    if source_cols != PEER_COLUMNS or peer_cols != PEER_COLUMNS:
        raise ValueError('Backfill or existing peer CSV has unexpected nine-column schema')
    if not {'challenge_id','settlement_date','official_deadline_utc'}.issubset(official_cols):
        raise ValueError('Official settlements CSV lacks expected columns')
    if len(source_rows)!=20:
        raise ValueError(f'Expected exactly 20 vetted backfill rows; found {len(source_rows)}')
    original_coverage = {(r['challenge_id'],r['agent_id']):r for r in peer_rows}
    if len(original_coverage)!=len(peer_rows):
        raise ValueError('Existing peer CSV already has duplicated challenge/agent IDs')
    official_by_id={r['challenge_id']:r for r in official_rows}
    if len(official_by_id)!=len(official_rows):
        raise ValueError('Official settlements CSV has duplicate challenge IDs')
    added_deadlines=[]
    for challenge,(date,cutoff) in EXPECTED_DEADLINES.items():
        r=official_by_id.get(challenge)
        if r is None:
            raise ValueError(f'Official settlement row missing: {challenge}')
        if r['settlement_date']!=date:
            raise ValueError(f'Incorrect challenge/date mapping: {challenge}')
        if r['official_deadline_utc'].strip():
            if as_utc(r['official_deadline_utc'])!=as_utc(cutoff):
                raise ValueError(f'Deadline conflict for {date}; source file is authoritative: STOP')
        else:
            r['official_deadline_utc']=cutoff
            added_deadlines.append(date)
    new={}
    for r in source_rows:
        cid=r['challenge_id']; aid=r['agent_id']; key=(cid,aid)
        if cid not in EXPECTED_DEADLINES:
            raise ValueError(f'Backfill contains unexpected challenge: {cid}')
        if aid==OWN_AGENT_ID:
            raise ValueError('Our own model must not appear in peer CSV')
        if key in new:
            raise ValueError(f'Duplicate backfill entry: {key}')
        try:
            probs=[Decimal(r[k]) for k in ('p_bearish','p_neutral','p_bullish')]
        except InvalidOperation:
            raise ValueError(f'Non-numeric peer probability: {key}')
        if any(not v.is_finite() or not 0<=v<=1 for v in probs) or sum(probs)!=Decimal('1'):
            raise ValueError(f'Invalid three-class probability vector: {key} {probs}')
        if r['counts_for_score'].lower()!='true':
            raise ValueError(f'Unsupported scoring eligibility flag: {key}')
        deadline=as_utc(EXPECTED_DEADLINES[cid][1]); submitted=as_utc(r['submitted_at_utc'])
        if submitted>=deadline:
            raise ValueError(f'Submission at/after official deadline: {key}')
        if r['source_url']!=f'https://headlinearena.com/prediction/{cid}':
            raise ValueError(f'Source URL does not match challenge: {key}')
        if not r['agent_name'].strip():
            raise ValueError(f'Missing peer name: {key}')
        new[key]=r
    additions=[];identical=0
    for key,r in new.items():
        if key in original_coverage:
            if r!=original_coverage[key]:
                raise ValueError(f'Existing record conflicts with backfill; refusing overwrite: {key}')
            identical+=1
        else:
            additions.append(r)
    print('EXISTING PEER ROWS:',len(peer_rows))
    print('BACKFILL ROWS VALIDATED:',len(source_rows))
    print('NEW PEER ROWS:',len(additions))
    print('ALREADY PRESENT, IDENTICAL:',identical)
    print('MISSING OFFICIAL DEADLINES TO FILL:',', '.join(added_deadlines) if added_deadlines else 'none')
    print('FINAL PEER ROWS:',len(peer_rows)+len(additions))
    if not args.apply:
        print('DRY RUN PASS: no files changed. Rerun with --apply to save.')
        return
    if not additions and not added_deadlines:
        print('ALREADY UP TO DATE: nothing changed.')
        return
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    # Never overwrite earlier backups.
    for source in (peer_file,official_file):
        backup=source.with_name(source.stem+f'.before_peer_backfill_{suffix}'+source.suffix)
        shutil.copy2(source,backup)
        print('BACKUP:',backup)
    if added_deadlines:
        write_atomic(official_file, official_cols, official_rows)
    if additions:
        write_atomic(peer_file, peer_cols, peer_rows+additions)
    _,check_peers=read_csv(peer_file)
    _,check_official=read_csv(official_file)
    if len(check_peers)!=len(peer_rows)+len(additions):
        raise RuntimeError('Post-write peer count mismatch')
    for cid in EXPECTED_DEADLINES:
        assert any(x['challenge_id']==cid and x['official_deadline_utc'] for x in check_official)
    print('APPLY PASS: updates saved and verified. Previous CSVs backed up.')

if __name__=='__main__':
    main()
