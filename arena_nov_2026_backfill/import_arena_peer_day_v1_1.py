#!/usr/bin/env python3
"""Audited, offline, general-purpose Headline Arena ES daily peer importer (revision-aware v1.1).

Designed for the original Market-Mood-Forecasting research schema, without
modifying model artifacts, predictions, submission receipts or notebooks.

Requirements: Python >= 3.10; Python standard library only.

Typical usage from project root, with actual challenge/date/hash from evidence:

  python import_arena_peer_day.py --source "data/raw/arena_peer_evidence/2026-10-30/arena_es_20261030.html" --date 2026-10-30 --challenge-id ACTUAL_UUID --inspect
  python import_arena_peer_day.py --source "..." --date YYYY-MM-DD --challenge-id ACTUAL_UUID
  python import_arena_peer_day.py --source "..." --date YYYY-MM-DD --challenge-id ACTUAL_UUID --expected-sha256 EXACT_HASH --apply

Without --apply, the tool never creates or modifies files. --inspect parses
only evidence, without requiring existing research CSVs. --apply requires an
explicit checksum pin and validates the source, schemas, chronological audit
chain, unique IDs, peer vectors, deadlines, audit cardinality and conflicts.

Supported source: complete saved *rendered/streamed Headline Arena HTML* that
exposes all scored prediction cards, with original three-class distributions
when supplied. It resolves React streaming S:*/P:* placeholders using the
saved DOM. It deliberately refuses unsupported layouts and uncertain evidence.
It DOES NOT fetch network data, reverse engineer missing probabilities, or
submit any forecasts. Do not use with an unreviewed/partial source page.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid

OUR_AGENT = 'agt_f7b9b0827875'
RESEARCH_DIR = Path('data/processed/es_arena_research_v0_1')
PEER_COLS = [
    'challenge_id', 'agent_id', 'agent_name', 'p_bearish', 'p_neutral',
    'p_bullish', 'submitted_at_utc', 'counts_for_score', 'source_url',
]
AUDIT_COLS = [
    'challenge_date', 'challenge_id', 'agent_id', 'agent_name', 'direction',
    'arena_score_card', 'display_bearish_pct', 'display_neutral_pct',
    'display_bullish_pct', 'display_sum_pct', 'prediction_card_timestamp_as_utc',
    'official_deadline_utc', 'probability_source', 'benchmark_status',
    'source_url', 'precision_note',
]
OFFICIAL_REQUIRED = {
    'challenge_id', 'settlement_date', 'asset', 'contract_symbol',
    'official_label', 'settlement_return_pct', 'official_deadline_utc',
    'official_settled_at_utc', 'arena_score', 'source_url', 'verified_by',
    'verified_at_utc',
}
VOID = {'area','base','br','col','embed','hr','img','input','link','meta',
        'param','source','track','wbr'}
ISO_CARD_RE = re.compile(
    r'(agt_[A-Za-z0-9]+)-(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)'
)
VEC_RE = re.compile(
    r'^Full probability distribution:\s*Bullish\s+(\d{1,3})%\s*·\s*'
    r'Neutral\s+(\d{1,3})%\s*·\s*Bearish\s+(\d{1,3})%$'
)
COMBINED_RE = re.compile(r'^es_peer_source_audit_(\d{8})_(\d{2}|\d{8})\.csv$')


class EvidenceError(ValueError):
    """Evidence is incomplete, inconsistent, or not proven by the source."""


@dataclass
class Element:
    tag: str
    attrs: dict[str, str | None] = field(default_factory=dict)
    children: list[Element | str] = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Element):
                yield from child.walk()

    def nodes(self, tag: str | None = None, **attrs):
        return [n for n in self.walk() if (tag is None or n.tag == tag)
                and all(n.attrs.get(k) == v for k, v in attrs.items())]

    def text(self):
        return ''.join(x.text() if isinstance(x, Element) else x for x in self.children)

    def has_class(self, name: str) -> bool:
        return name in (self.attrs.get('class') or '').split()


class TreeParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Element('document')
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        item = Element(tag, dict(attrs))
        self.stack[-1].children.append(item)
        if tag not in VOID:
            self.stack.append(item)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Element(tag, dict(attrs)))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def one(items, what: str):
    if len(items) != 1:
        raise EvidenceError(f'Expected exactly one {what}, found {len(items)}')
    return items[0]


def utc(text: str) -> datetime:
    raw = text.strip()
    if raw.endswith('Z'):
        raw = raw[:-1] + '+00:00'
    if not re.search(r'[+-]\d\d:\d\d$', raw):
        # Source's React card keys are platform UTC timestamps without offset.
        raw += '+00:00'
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        raise EvidenceError(f'Naive timestamp: {text}')
    return value.astimezone(timezone.utc)


def pretty_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def parse_number(s: str) -> float:
    value = float(s.replace(',', '').replace('%', '').replace('pt', '').strip())
    if not math.isfinite(value):
        raise EvidenceError(f'Nonfinite numeric value: {s}')
    return value


def inflate_streamed_html(root: Element) -> Element:
    """Reconstruct the saved SSR detail page without executing browser JS.

    React/Next streaming snapshots have a primary S:2 container and later
    matching S:n fragments whose contents replace P:n template placeholders.
    Only the detail page subtree is expanded, avoiding duplicated fragment cards.
    """
    by_id = {n.attrs.get('id'): n for n in root.walk() if n.attrs.get('id')}
    primary = by_id.get('S:2')
    if primary is None:
        # Fully rendered static HTML is also valid.
        pages = [n for n in root.nodes('main') if n.has_class('detail-main')]
        return copy.deepcopy(one(pages, 'rendered detail page'))
    page = copy.deepcopy(primary)

    def expand(node: Element, depth: int = 0):
        if depth > 70:
            raise EvidenceError('Excessive streamed template nesting')
        output = []
        for child in node.children:
            if isinstance(child, Element) and child.tag == 'template' and (child.attrs.get('id') or '').startswith('P:'):
                fragment_id = 'S:' + child.attrs['id'][2:]
                fragment = by_id.get(fragment_id)
                if fragment is None:
                    # Some templates relate to metadata or offscreen React scaffolding.
                    continue
                clone = copy.deepcopy(fragment)
                expand(clone, depth + 1)
                output.extend(clone.children)
            else:
                if isinstance(child, Element):
                    expand(child, depth + 1)
                output.append(child)
        node.children = output

    expand(page)
    return one([n for n in page.nodes('main') if n.has_class('detail-main')], 'streamed detail page')


def entry_from_dt(main: Element, label: str) -> Element:
    items = []
    for dl in main.nodes('dl'):
        for child in dl.children:
            if not isinstance(child, Element) or child.tag != 'div':
                continue
            dt = [n for n in child.children if isinstance(n, Element) and n.tag == 'dt']
            dd = [n for n in child.children if isinstance(n, Element) and n.tag == 'dd']
            if len(dt) == 1 and len(dd) == 1 and dt[0].text().strip() == label:
                items.append(dd[0])
    return one(items, f'official detail field {label!r}')


def parse_evidence(source: Path, cid: str, day: date, expected_sha: str | None):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha and digest.lower() != expected_sha.lower():
        raise EvidenceError(f'SOURCE SHA256 MISMATCH: {digest} (expected {expected_sha})')
    html = raw.decode('utf-8-sig')
    parser = TreeParser()
    parser.feed(html)
    root = parser.root
    url = f'https://headlinearena.com/prediction/{cid}'
    canonical = [n.attrs.get('href') for n in root.nodes('link')
                 if n.attrs.get('rel') == 'canonical']
    if canonical != [url]:
        raise EvidenceError(f'Canonical URL mismatch: {canonical!r}; expected {url}')
    main = inflate_streamed_html(root)
    resolved = one([n for n in main.nodes('span') if n.has_class('status-chip')], 'status chip')
    if resolved.attrs.get('data-status') != 'resolved':
        raise EvidenceError('Challenge has not been officially resolved')

    marked_asset = [n for n in main.nodes('p') if n.has_class('asset-mark')]
    if len(marked_asset) != 1 or not re.search(r'\bES\b', marked_asset[0].text()):
        raise EvidenceError('Not an identified ES challenge')
    contract_nodes = [n for n in main.nodes('code') if re.fullmatch(r'ES[A-Z]\d', n.text().strip())]
    contract = one(contract_nodes, 'frozen ES contract').text().strip()
    if contract != 'ESZ6':
        # Contract roll can be supported without training changes only after
        # separately validating model contract policy. This importer stores
        # evidence but does not infer on it.
        print(f'NOTE: challenge contract {contract}; verify contract roll policy.', file=sys.stderr)

    deadline_el = entry_from_dt(main, 'Predictions close')
    deadline = utc(one(deadline_el.nodes('time'), 'deadline timestamp').attrs['datetime'])
    resolution_el = entry_from_dt(main, 'Resolved')
    approximate_resolved = utc(one(resolution_el.nodes('time'), 'resolution timestamp').attrs['datetime'])
    official_resolution_date = approximate_resolved.date()
    if day != deadline.date() or day != official_resolution_date:
        raise EvidenceError(f'Provided date {day} does not match deadline/resolution date')

    datasets = []
    for script in root.nodes('script'):
        if script.attrs.get('type') != 'application/ld+json':
            continue
        try:
            data = json.loads(script.text())
        except (ValueError, TypeError):
            continue
        datasets += [x for x in data.get('@graph', []) if x.get('@type') == 'Dataset'
                     and x.get('url') == url]
    dataset = one(datasets, 'JSON-LD dataset for this exact challenge')
    resolved_at = utc(dataset['dateModified'])
    if abs((resolved_at - approximate_resolved).total_seconds()) >= 1.0:
        raise EvidenceError('Source JSON-LD resolution and displayed resolution disagree')

    settle_dd = entry_from_dt(main, 'Settlement price')
    settlement_price = parse_number(settle_dd.text())
    market_changes = [n for n in main.nodes('span') if n.has_class('market-change')]
    market_change = one(market_changes, 'official settled market return')
    market_return = parse_number(market_change.text())
    outcome = one([n for n in main.nodes('strong') if n.text().strip().lower()
                   in ('bearish', 'neutral', 'bullish')], 'official outcome').text().strip().lower()
    opens = []
    for span in main.nodes('span'):
        text = span.text()
        m = re.fullmatch(r'\s*Open price\s*:\s*([\d,]+(?:\.\d+)?)\s*', text)
        if m:
            opens.append(parse_number(m.group(1)))
    opening_price = one(opens, 'official recorded opening price')
    if opening_price <= 0 or settlement_price <= 0:
        raise EvidenceError('Non-positive official prices')
    derived_return = 100 * (settlement_price / opening_price - 1)
    if abs(derived_return - market_return) > 0.00011:
        raise EvidenceError(f'Opening/settlement price not consistent with reported %: {derived_return} vs {market_return}')
    implied = 'bearish' if market_return < -0.3 else ('bullish' if market_return > 0.3 else 'neutral')
    if outcome != implied:
        raise EvidenceError('Official direction does not match ±0.3% rule')
    if 'Moves beyond ±0.3%' not in main.text():
        raise EvidenceError('Expected ±0.30% dead zone is not documented in source')
    participants = int(entry_from_dt(main, 'Predictions').text().strip())

    card_keys = set(ISO_CARD_RE.findall(html))
    if len(card_keys) != participants:
        raise EvidenceError(f'Expected {participants} unique scored prediction timestamp keys, found {len(card_keys)}')
    agent_times: dict[str, str] = {}
    for aid, stamp in card_keys:
        if aid in agent_times:
            raise EvidenceError(f'Multiple prediction timestamp keys for agent {aid}; manual revision audit required')
        agent_times[aid] = stamp
    articles = [n for n in main.nodes('article') if n.has_class('prediction-card')]
    if len(articles) != participants:
        raise EvidenceError(f'Prediction cards incomplete: expected {participants}, reconstructed {len(articles)}')

    agents = []
    for card in articles:
        links = [n for n in card.nodes('a') if re.fullmatch(r'/agent/agt_[A-Za-z0-9]+', n.attrs.get('href') or '')]
        link = one(links, 'card agent identifier/name')
        aid = link.attrs['href'].split('/')[-1]
        name = link.text().strip()
        if not name or aid not in agent_times:
            raise EvidenceError(f'Missing agent name or timestamp key for {aid}')
        stamp = utc(agent_times[aid])
        if not stamp < deadline:
            raise EvidenceError(f'Late/equal-deadline agent submission: {aid}')
        directions = [n for n in card.nodes('span') if n.has_class('direction-badge')]
        direction = one(directions, f'direction for {aid}').attrs.get('data-direction')
        if direction not in ('bearish','neutral','bullish'):
            raise EvidenceError(f'Unexpected direction for {aid}: {direction}')
        score_chip = one([n for n in card.nodes('span') if n.has_class('score-chip')], f'scored result for {aid}')
        score_text = ' '.join(score_chip.text().split())
        match_score = re.fullmatch(r'Score\s+(\d+)\s*·\s*(correct|missed)', score_text, re.I)
        if not match_score:
            raise EvidenceError(f'Scored status ambiguous for {aid}: {score_text!r}')
        score, correctness = int(match_score.group(1)), match_score.group(2).lower()
        if (correctness == 'correct') != (direction == outcome):
            raise EvidenceError(f'Scored correctness contradicts official outcome for {aid}')
        # Only the top-level scored card's probability distribution is its
        # current publicly displayed vector. Any distribution-strip inside a
        # revision-log is historical, *not* an additional scored forecast.
        # Preserve the original source and audit existence of the revisions.
        all_dist_nodes = [n for n in card.walk()
                          if (n.attrs.get('aria-label') or '').startswith('Full probability distribution:')]
        current_nodes = [n for n in card.walk()
                         if n.has_class('confidence-bar--distribution')
                         and (n.attrs.get('aria-label') or '').startswith('Full probability distribution:')]
        if len(current_nodes) > 1:
            raise EvidenceError(f'Ambiguous current scored-card distribution for {aid}: {len(current_nodes)}')
        revision_logs = [n for n in card.nodes('details') if n.has_class('revision-log')]
        if len(revision_logs) > 1:
            raise EvidenceError(f'Ambiguous multiple revision-history sections for {aid}')
        historical_nodes = []
        revision_count = 0
        for history in revision_logs:
            summary = one([n for n in history.children if isinstance(n, Element) and n.tag == 'summary'],
                          f'revision-history summary for {aid}')
            count_match = re.fullmatch(r'Revision history \((\d+)\)', ' '.join(summary.text().split()))
            if not count_match:
                raise EvidenceError(f'Unexpected revision-history summary for {aid}: {summary.text()!r}')
            revision_count = int(count_match.group(1))
            revision_lists = [n for n in history.children if isinstance(n, Element) and n.tag == 'ul']
            entries = [n for item in revision_lists for n in item.children if isinstance(n, Element) and n.tag == 'li']
            if len(entries) != revision_count:
                raise EvidenceError(f'Historical revision entries for {aid}: expected {revision_count}, found {len(entries)}')
            for entry in entries:
                hist_time = one(entry.nodes('time'), f'historical revision time for {aid}').attrs.get('datetime')
                if not hist_time or not utc(hist_time) < deadline:
                    raise EvidenceError(f'Historical revision timestamp invalid or late for {aid}: {hist_time}')
                entry_dist = [n for n in entry.walk()
                              if (n.attrs.get('aria-label') or '').startswith('Full probability distribution:')]
                if len(entry_dist) > 1:
                    raise EvidenceError(f'Multiple historical distributions in one revision for {aid}')
                for n in entry_dist:
                    label = n.attrs['aria-label']
                    parsed = VEC_RE.fullmatch(label)
                    if not parsed:
                        raise EvidenceError(f'Unsupported historical distribution for {aid}: {label!r}')
                    nums = tuple(map(int, parsed.groups()))
                    if any(not 0 <= x <= 100 for x in nums) or not 98 <= sum(nums) <= 102:
                        raise EvidenceError(f'Out-of-range historical probability vector for {aid}: {label!r}')
                    historical_nodes.append(n)
        accounted_for = {id(n) for n in current_nodes + historical_nodes}
        if any(id(n) not in accounted_for for n in all_dist_nodes):
            raise EvidenceError(f'Unclassified probability distribution outside scored card/revision history for {aid}')
        # Some agents submitted only a direction and confidence, without a
        # full vector. Preserve that case; never invent missing probabilities.
        probs = None
        if current_nodes:
            vec = current_nodes[0].attrs['aria-label']
            m = VEC_RE.fullmatch(vec)
            if not m:
                raise EvidenceError(f'Unsupported current scored-card distribution for {aid}: {vec!r}')
            bull, neutral, bear = map(int, m.groups())
            probs = (bear, neutral, bull)
            if any(not (0 <= x <= 100) for x in probs):
                raise EvidenceError(f'Invalid displayed probability for {aid}')
            maximum = max(probs)
            index = ('bearish', 'neutral', 'bullish').index(direction)
            if probs[index] != maximum:
                raise EvidenceError(f'Highest current probability does not match scored direction for {aid}')
        agents.append({'agent_id':aid,'agent_name':name,'direction':direction,
                       'score':score,'correctness':correctness,
                       'stamp':pretty_utc(stamp),'pct':probs,
                       'revision_count':revision_count,
                       'historical_vectors':len(historical_nodes)})
    if len(agents) != len({a['agent_id'] for a in agents}):
        raise EvidenceError('Repeated agent IDs/revised cards in source; manual review required')
    if set(agent_times) != {a['agent_id'] for a in agents}:
        raise EvidenceError('Prediction cards/timestamps inconsistent')
    agents.sort(key=lambda a: (a['stamp'], a['agent_id']))
    return {'sha256':digest,'url':url,'day':day.isoformat(),'id':cid,
            'contract':contract,'deadline':pretty_utc(deadline),
            'resolved_at':pretty_utc(resolved_at),'opening_price':opening_price,
            'settlement_price':settlement_price,'settlement_return_pct':market_return,
            'official_label':outcome,'participants':participants,'agents':agents}


def evidence_rows(evidence: dict):
    peers, audit = [], []
    for a in evidence['agents']:
        own = (a['agent_id'] == OUR_AGENT)
        pct = a['pct']
        total = sum(pct) if pct is not None else None
        if own:
            status = 'excluded_our_own_model'
        elif pct is None:
            status = 'excluded_no_original_vector'
        elif total != 100:
            status = f'excluded_display_rounding_sum_{total}'
        else:
            status = 'included_peer_csv'
        if pct is None:
            precision = 'No original three-class vector exposed'
        else:
            precision = 'Rounded whole-percent latest scored-card probabilities; unrounded submitted vector not verified'
        if a.get('revision_count',0):
            precision += (f'; revision history ({a["revision_count"]}) exists, '
                          'historical distributions retained exclusively in immutable source; '
                          'not treated as distinct scored forecasts')
        audit.append(dict(zip(AUDIT_COLS,[
            evidence['day'], evidence['id'], a['agent_id'], a['agent_name'],
            a['direction'].capitalize(), f"Score {a['score']} · {a['correctness']}",
            *([str(float(x)) for x in pct] if pct else ['','','']),
            str(float(total)) if total is not None else '', a['stamp'],
            evidence['deadline'], 'rendered HTML' if pct else 'not_rendered',
            status, evidence['url'], precision,
        ])))
        if status == 'included_peer_csv':
            peers.append(dict(zip(PEER_COLS,[
                evidence['id'], a['agent_id'], a['agent_name'],
                *[str(x / 100) for x in pct], a['stamp'], 'True', evidence['url'],
            ])))
    assert len(audit) == evidence['participants']
    assert all(r['agent_id'] != OUR_AGENT for r in peers)
    return peers, audit


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Expected existing CSV missing: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        cols = reader.fieldnames
        records = list(reader)
    if not cols or len(cols) != len(set(cols)) or any(None in row or None in row.values() for row in records):
        raise EvidenceError(f'Malformed CSV (headers or row length): {path}')
    if any(set(row) != set(cols) for row in records):
        raise EvidenceError(f'Malformed CSV keys: {path}')
    return cols, records


def unique_key(rows, key_fields, label):
    keys = [tuple(r[k] for k in key_fields) for r in rows]
    if len(keys) != len(set(keys)):
        raise EvidenceError(f'Duplicate {label} keys in existing evidence')


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_enddate(file: Path):
    m = COMBINED_RE.fullmatch(file.name)
    if not m:
        return None
    start = date.fromisoformat(m.group(1)[:4]+'-'+m.group(1)[4:6]+'-'+m.group(1)[6:])
    suffix = m.group(2)
    end = date(int(suffix[:4]),int(suffix[4:6]),int(suffix[6:8])) if len(suffix) == 8 else date(start.year,start.month,int(suffix))
    if end < start:
        raise EvidenceError(f'Invalid combined audit chronology: {file}')
    return start, end


def audit_chain(base: Path, day: date):
    combined = []
    for path in base.glob('es_peer_source_audit_*.csv'):
        parsed = audit_enddate(path)
        if parsed is not None:
            combined.append((parsed[0], parsed[1], path))
    if not combined:
        raise EvidenceError('No existing dated consolidated audit; initialize with verified history first')
    start_dates = {x[0] for x in combined}
    if len(start_dates) != 1:
        raise EvidenceError('Conflicting consolidated audit start dates')
    start = next(iter(start_dates))
    if day < start:
        raise EvidenceError('Target date predates historical audit start')
    latest_end = max(e for _, e, _ in combined)
    latest_path = one([p for _, e, p in combined if e == latest_end], 'latest consolidated audit')
    if day < latest_end:
        raise EvidenceError('Out-of-order historical import. Use --inspect, or rebuild audit chain in a separately reviewed migration')
    if day == latest_end:
        # The latest consolidated audit should already include this day.
        return latest_path, latest_path, True
    suffix = f'{day.day:02d}' if (day.year, day.month) == (start.year, start.month) else day.strftime('%Y%m%d')
    new_path = base / f'es_peer_source_audit_{start:%Y%m%d}_{suffix}.csv'
    if new_path.exists():
        raise EvidenceError(f'Conflicting consolidated audit file already exists: {new_path}')
    return latest_path, new_path, False


def compare_peer(old: dict, new: dict):
    for col in PEER_COLS:
        left, right = old[col].strip(), new[col].strip()
        if col.startswith('p_'):
            equal = abs(float(left) - float(right)) < 1e-12
        elif col == 'counts_for_score':
            equal = left.lower() == right.lower()
        else:
            equal = left == right
        if not equal:
            raise EvidenceError(f'Conflicting existing peer {old["challenge_id"]}/{old["agent_id"]}: {col}')


def compare_official(old: dict, new: dict):
    cols = ['challenge_id','settlement_date','asset','contract_symbol','official_label',
            'settlement_return_pct','official_deadline_utc','official_settled_at_utc',
            'arena_score','source_url']
    for col in cols:
        left, right = old[col].strip(), new[col].strip()
        if col == 'settlement_return_pct':
            equal = abs(float(left) - float(right)) < 1e-10
        elif col in ('official_deadline_utc','official_settled_at_utc'):
            equal = utc(left) == utc(right)
        else:
            equal = left == right
        if not equal:
            raise EvidenceError(f'Conflicting existing official outcome {old["challenge_id"]}: {col}')


def validate_existing_day_audit(existing: list[dict], proposed: list[dict]):
    if len(existing) != len(proposed):
        raise EvidenceError('Existing daily audit has a different participant count')
    by_agent = {r['agent_id']:r for r in existing}
    if len(by_agent) != len(existing):
        raise EvidenceError('Existing daily audit has duplicated agent IDs')
    # Allow differing wording in historical precision_note, while preserving
    # the original source and scientifically meaningful fields verbatim.
    for row in proposed:
        prior = by_agent.get(row['agent_id'])
        if prior is None:
            raise EvidenceError(f'Previously recorded daily audit missing agent {row["agent_id"]}')
        for k in AUDIT_COLS:
            if k in ('precision_note',):
                continue
            if prior[k].strip() != row[k].strip():
                raise EvidenceError(f'Existing daily audit conflict {row["agent_id"]}: {k}')


def plan_import(evidence:dict, base:Path):
    peers_new, day_audit = evidence_rows(evidence)
    peer_path = base / 'arena_peer_forecasts.csv'
    official_path = base / 'arena_official_settlements.csv'
    day_file = base / f'es_peer_source_audit_{evidence["day"].replace("-", "")}.csv'
    pcols, peers = read_csv(peer_path)
    ocols, officials = read_csv(official_path)
    if pcols != PEER_COLS:
        raise EvidenceError(f'Unexpected peer schema: {pcols}')
    if not OFFICIAL_REQUIRED.issubset(ocols):
        raise EvidenceError('Unexpected official settlements schema')
    unique_key(peers, ('challenge_id','agent_id'), 'peer')
    unique_key(officials, ('challenge_id',), 'official')
    if any(r['agent_id'] == OUR_AGENT for r in peers):
        raise EvidenceError('Own model present in existing peer dataset')
    prev_file, combined_file, same_day = audit_chain(base, date.fromisoformat(evidence['day']))
    acols, old_audit = read_csv(prev_file)
    if acols != AUDIT_COLS:
        raise EvidenceError('Unexpected historical audit schema')
    unique_key(old_audit, ('challenge_id','agent_id'), 'historical audit')
    if day_file.exists():
        dcols, drows = read_csv(day_file)
        if dcols != AUDIT_COLS:
            raise EvidenceError('Unexpected daily audit schema')
        validate_existing_day_audit(drows, day_audit)
    else:
        drows = None
    if same_day:
        if drows is None:
            raise EvidenceError('Latest consolidated audit includes this day, but daily audit is missing')
        existing_this_day = [r for r in old_audit if r['challenge_id'] == evidence['id']]
        validate_existing_day_audit(existing_this_day, day_audit)
        new_combined = old_audit
    else:
        if any(r['challenge_id'] == evidence['id'] for r in old_audit):
            raise EvidenceError('Challenge already in prior historical combined audit')
        if any(r['challenge_date'] >= evidence['day'] for r in old_audit):
            raise EvidenceError('Previous audit has sessions at or after target date')
        new_combined = old_audit + day_audit

    previous_peers = {(r['challenge_id'],r['agent_id']):r for r in peers}
    additions = []
    for row in peers_new:
        key = (row['challenge_id'],row['agent_id'])
        if key in previous_peers:
            compare_peer(previous_peers[key],row)
        else:
            additions.append(row)
    # Refuse incomplete or duplicate peer records for already imported day.
    if same_day:
        full_set = {(r['challenge_id'],r['agent_id']) for r in peers_new}
        actual_set = {(r['challenge_id'],r['agent_id']) for r in peers if r['challenge_id'] == evidence['id']}
        if full_set != actual_set:
            raise EvidenceError('Existing imported day has incomplete/excess peer coverage')

    own_score = next((str(a['score']) for a in evidence['agents'] if a['agent_id'] == OUR_AGENT), '')
    official_new = dict.fromkeys(ocols, '')
    official_new.update(
        challenge_id=evidence['id'],settlement_date=evidence['day'],
        asset='ES',contract_symbol=evidence['contract'],
        official_label=evidence['official_label'],
        settlement_return_pct=str(evidence['settlement_return_pct']),
        official_deadline_utc=evidence['deadline'],
        official_settled_at_utc=evidence['resolved_at'],
        arena_score=own_score,source_url=evidence['url'],
        verified_by='user_supplied_arena_html_sha256:'+evidence['sha256'],
        verified_at_utc=pretty_utc(datetime.now(timezone.utc)),
    )
    old_official = next((r for r in officials if r['challenge_id'] == evidence['id']),None)
    if old_official:
        compare_official(old_official,official_new)
    elif same_day:
        raise EvidenceError('Combined audit already includes day, but official settlement is missing')

    writes = {}
    if additions:
        writes[peer_path] = (pcols, peers + additions)
    if old_official is None:
        writes[official_path] = (ocols, officials + [official_new])
    if drows is None:
        writes[day_file] = (AUDIT_COLS,day_audit)
    if not same_day:
        writes[combined_file] = (AUDIT_COLS,new_combined)
    return {'writes':writes,'peer_path':peer_path,'official_path':official_path,
            'audit_path':day_file,'combined_path':combined_file,
            'old_peers':peers,'old_officials':officials,
            'old_peer_sha256':file_digest(peer_path),
            'old_official_sha256':file_digest(official_path),
            'prior_audit_path':prev_file,
            'prior_audit_sha256':file_digest(prev_file),
            'added_peers':len(additions),'added_official':int(old_official is None),
            'old_audit_count':len(old_audit),'combined_count':len(new_combined),
            'daily_count':len(day_audit),'eligible_peer_count':len(peers_new)}


def render_csv_bytes(cols, rows) -> bytes:
    import io
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out,fieldnames=cols,lineterminator='\n',extrasaction='raise')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8')


def execute_apply(plan:dict, evidence:dict, base:Path):
    writes = plan['writes']
    if not writes:
        print('ALREADY UP TO DATE; no files changed.')
        return
    lock = base/'.arena_peer_import.lock'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise EvidenceError(f'Import lock exists: {lock}. Check for concurrent/crashed import before removing it.') from exc
    os.close(fd)
    temps = {}
    backups = {}
    new_files = []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        # Check before touching any canonical: existing files are the same ones
        # used by the dry-run plan (prevent accidental concurrent changes).
        for path in writes:
            if path.exists() and path not in (plan['peer_path'],plan['official_path']):
                raise EvidenceError(f'Refusing to replace preexisting immutable audit: {path}')
        if file_digest(plan['peer_path']) != plan['old_peer_sha256'] or file_digest(plan['official_path']) != plan['old_official_sha256']:
            raise EvidenceError('Canonical CSV changed since validation; rerun dry-run')
        if file_digest(plan['prior_audit_path']) != plan['prior_audit_sha256']:
            raise EvidenceError('Previous combined audit changed since validation; rerun dry-run')
        original_hashes = {plan['peer_path']:plan['old_peer_sha256'],
                           plan['official_path']:plan['old_official_sha256']}
        for path, (cols,rows) in writes.items():
            content = render_csv_bytes(cols,rows)
            with tempfile.NamedTemporaryFile(dir=base,prefix='.arena_peer_stage_',suffix='.tmp',delete=False) as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
                temps[path] = Path(tmp.name)
        for path in (plan['peer_path'],plan['official_path']):
            if file_digest(path) != original_hashes[path]:
                raise EvidenceError(f'Unexpected external changes to canonical {path}')
            backup = path.with_name(path.stem + '.before_' + evidence['day'].replace('-','') + '_' + stamp + path.suffix)
            if backup.exists():
                raise EvidenceError(f'Backup collision: {backup}')
            shutil.copy2(path,backup)
            backups[path] = backup
            print('BACKUP:',backup)
        for path, staged in temps.items():
            if path.exists() and path not in backups:
                raise EvidenceError(f'Unexpected destination file present: {path}')
            if not path.exists():
                new_files.append(path)
            os.replace(staged,path)
        # Re-read and independently assert all expected outcomes.
        _, p = read_csv(plan['peer_path'])
        _, o = read_csv(plan['official_path'])
        _, daily = read_csv(plan['audit_path'])
        _, combined = read_csv(plan['combined_path'])
        unique_key(p,('challenge_id','agent_id'),'postwrite peer')
        unique_key(o,('challenge_id',),'postwrite settlement')
        unique_key(combined,('challenge_id','agent_id'),'postwrite audit')
        if any(x['agent_id'] == OUR_AGENT for x in p):
            raise EvidenceError('Own agent found in peer dataset after import')
        if len(p) != len(plan['old_peers']) + plan['added_peers']:
            raise EvidenceError('Peer row count mismatch after import')
        if len(o) != len(plan['old_officials']) + plan['added_official']:
            raise EvidenceError('Official row count mismatch after import')
        if len(daily) != plan['daily_count'] or len(combined) != plan['combined_count']:
            raise EvidenceError('Audit row count mismatch after import')
        if {(x['challenge_id'],x['agent_id']) for x in p if x['challenge_id']==evidence['id']} != {
            (x['challenge_id'],x['agent_id']) for x in evidence_rows(evidence)[0]}:
            raise EvidenceError('Peer identity coverage mismatch after import')
        if sum(x['challenge_id']==evidence['id'] for x in o) != 1:
            raise EvidenceError('Official challenge count mismatch after import')
        if sum(x['challenge_id']==evidence['id'] for x in combined) != evidence['participants']:
            raise EvidenceError('Audit participant count mismatch after import')
        print('APPLY VERIFICATION PASS — frozen model/forecast files were not touched.')
    except Exception:
        for path, backup in backups.items():
            shutil.copy2(backup,path)
        for path in new_files:
            path.unlink(missing_ok=True)
        print('APPLICATION FAILED; restored existing canonicals from backups. Investigate.',file=sys.stderr)
        raise
    finally:
        for staged in temps.values():
            staged.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source',type=Path,required=True,help='Locally saved rendered/streamed official challenge HTML')
    parser.add_argument('--date',required=True,help='Verified settlement date YYYY-MM-DD')
    parser.add_argument('--challenge-id',required=True,help='Exact Arena challenge UUID, verified independently')
    parser.add_argument('--research-dir',type=Path,default=RESEARCH_DIR,help='Existing private CSV research directory')
    parser.add_argument('--expected-sha256',help='Expected SHA256 of exact locally reviewed HTML; mandatory for --apply')
    parser.add_argument('--inspect',action='store_true',help='Evidence parsing only; never touches research CSVs')
    parser.add_argument('--apply',action='store_true',help='Explicitly authorize research CSV changes after backups')
    parser.add_argument('--acknowledge-no-peer-vectors',action='store_true',
                        help='Allow an audited day with competitors but zero visible valid peer vectors')
    args = parser.parse_args(argv)
    if args.inspect and args.apply:
        parser.error('--inspect cannot be combined with --apply')
    if args.apply and (not args.expected_sha256 or not re.fullmatch(r'[0-9a-fA-F]{64}',args.expected_sha256)):
        parser.error('--apply requires --expected-sha256 with a verified 64-digit hexadecimal hash')
    if args.expected_sha256 and not re.fullmatch(r'[0-9a-fA-F]{64}',args.expected_sha256):
        parser.error('Invalid --expected-sha256')
    day = date.fromisoformat(args.date)
    try:
        cid = str(uuid.UUID(args.challenge_id))
    except ValueError as exc:
        parser.error(f'Invalid challenge UUID: {exc}')
    if cid != args.challenge_id.lower():
        parser.error('Challenge UUID must use the canonical hyphenated format')
    evidence = parse_evidence(args.source,cid,day,args.expected_sha256)
    peers, day_audit = evidence_rows(evidence)
    competitors = [a for a in evidence['agents'] if a['agent_id'] != OUR_AGENT]
    if competitors and not peers:
        print('WARNING: No eligible peer vectors were visible. Check signed-in page and disclosure before importing.',file=sys.stderr)
        if args.apply and not args.acknowledge_no_peer_vectors:
            raise EvidenceError('No eligible peer vectors: --apply requires --acknowledge-no-peer-vectors after manual investigation')
    print('SOURCE SHA256:',evidence['sha256'])
    print('CHALLENGE:',evidence['id'],'DATE:',evidence['day'],'CONTRACT:',evidence['contract'])
    print('OFFICIAL:',evidence['official_label'],f'{evidence["settlement_return_pct"]:.4f}%',
          'OPEN:',evidence['opening_price'],'SETTLE:',evidence['settlement_price'],
          'RESOLVED:',evidence['resolved_at'])
    print('DEADLINE:',evidence['deadline'])
    print('SCORED CARDS:',evidence['participants'],
          'OWN:',sum(x['agent_id']==OUR_AGENT for x in evidence['agents']),
          'ELIGIBLE PEERS:',len(peers))
    for x in evidence['agents']:
        print('  AGENT:',x['agent_id'],x['agent_name'],x['direction'],
              'Score',x['score'],'UTC:',x['stamp'],'PCT:',x['pct'],
              'HISTORICAL_REVISIONS:',x.get('revision_count',0),
              'HISTORICAL_VECTORS:',x.get('historical_vectors',0))
    print('INCLUSION STATUS:',{name:sum(r['benchmark_status']==name for r in day_audit)
                               for name in sorted(set(r['benchmark_status'] for r in day_audit))})
    if args.inspect:
        print('EVIDENCE INSPECTION PASS; no files changed.')
        return 0
    plan = plan_import(evidence,args.research_dir)
    print('PEER ROWS:',len(plan['old_peers']),'->',len(plan['old_peers'])+plan['added_peers'])
    print('SETTLEMENT ROWS:',len(plan['old_officials']),'->',len(plan['old_officials'])+plan['added_official'])
    print('DAILY AUDIT ROWS:',plan['daily_count'],'COMBINED AUDIT ROWS:',plan['old_audit_count'],'->',plan['combined_count'])
    print('DATE AUDIT:',plan['audit_path'])
    print('COMBINED AUDIT:',plan['combined_path'])
    if not args.apply:
        print('DRY RUN PASS; nothing changed. Review output and exact source SHA before --apply.')
    else:
        execute_apply(plan,evidence,args.research_dir)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (EvidenceError,OSError,ValueError,KeyError) as exc:
        print('STOP — VALIDATION ERROR:',exc,file=sys.stderr)
        sys.exit(2)
