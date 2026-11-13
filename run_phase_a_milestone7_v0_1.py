"""Re-run frozen public Notebook 12 in memory and build a private 18-settlement checkpoint.

Read-only with respect to repository evidence; writes only to a NEW private output
folder outside the repository. Requires nbclient, nbformat and the notebook's
existing notebook dependencies. Never performs a network request itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

CLASSES = ("bearish", "neutral", "bullish")
EXPECTED_NOV12_ID = "5638f558-5832-4fda-9bf5-d760a209917d"
DEFAULT_NOTEBOOK = "notebooks/12_es_arena_prospective_evaluation_v0_3_2_public.ipynb"
MARKER = "PHASE_A_M7_PAYLOAD_JSON="

# Executed IN the existing Notebook 12 kernel, after all existing cells.
# Pull the already qualified, full-precision data from its own eval_df.
EXPORT_CELL = r'''
import json as _m7_json
if "eval_df" not in globals() or "MODEL_METRICS" not in globals():
    raise RuntimeError("Notebook 12 evaluation variables missing")
_m7_cols = ["challenge_id", "settlement_date", "official_label", "predicted_label",
            "p_bearish", "p_neutral", "p_bullish"]
_m7_missing = [x for x in _m7_cols if x not in eval_df.columns]
if _m7_missing:
    raise RuntimeError("Notebook 12 missing columns: " + str(_m7_missing))
_m7_rows = []
for _, _m7_r in eval_df.sort_values(["settlement_date","challenge_id"]).iterrows():
    _m7_rows.append({
        "challenge_id": str(_m7_r["challenge_id"]),
        "settlement_date": str(_m7_r["settlement_date"]),
        "official_label": str(_m7_r["official_label"]),
        "predicted_label": str(_m7_r["predicted_label"]),
        "p_bearish": float(_m7_r["p_bearish"]),
        "p_neutral": float(_m7_r["p_neutral"]),
        "p_bullish": float(_m7_r["p_bullish"]),
    })
_m7_prior = prior.tolist() if prior is not None else None
_m7_payload = {
    "rows": _m7_rows,
    "notebook_metrics": {k: (float(v) if k != "n" else int(v))
                         for k,v in (MODEL_METRICS or {}).items()},
    "strict_native_settled_n": len(strict_eval_df),
    "accepted_evidenced_n": int(ledger.accepted_with_evidence.sum()) if not ledger.empty else 0,
    "official_settlement_rows": len(arena),
    "peer_rows": len(peer),
    "training_only_prior": _m7_prior,
    "notebook_study_status": quality_report.get("analysis_status", "UNKNOWN"),
    "excluded_settled_first30_n": int(full.excluded_from_descriptive_metrics.sum()) if not full.empty else 0,
}
print("PHASE_A_M7_PAYLOAD_JSON=" + _m7_json.dumps(_m7_payload, allow_nan=False, separators=(",",":")))
'''


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def _safe_outdir(out: Path, root: Path) -> Path:
    out = out.resolve()
    root = root.resolve()
    if out == root or root in out.parents:
        raise ValueError("Output directory must be OUTSIDE the Git repository")
    if out.exists():
        raise FileExistsError("Output directory already exists; choose a NEW path: " + str(out))
    if not out.parent.is_dir():
        raise ValueError("Output parent folder does not exist: " + str(out.parent))
    return out


def _score(q: list[list[float]], labels: list[str]) -> dict:
    n = len(labels)
    if not n or len(q) != n:
        raise ValueError("Empty or mismatched scored forecast set")
    idx = {c:i for i,c in enumerate(CLASSES)}
    correct = brier = logloss = confidence = 0.0
    for p, actual in zip(q, labels):
        a = idx[actual]
        y = [int(k == a) for k in range(3)]
        pred = max(range(3), key=lambda k:p[k])
        correct += pred == a
        brier += sum((p[k] - y[k])**2 for k in range(3))
        logloss += -math.log(max(1e-15, p[a]))
        confidence += max(p)
    return dict(n=n, accuracy=correct/n, brier=brier/n,
                log_loss=logloss/n, avg_confidence=confidence/n)


def compute_report(payload: dict, expected_n: int=18) -> dict:
    rows = payload.get('rows')
    if not isinstance(rows,list) or len(rows) != expected_n:
        raise ValueError(f"Expected {expected_n} eligible settled rows, found {len(rows) if isinstance(rows,list) else 'invalid'}; STOP")
    ids = [x['challenge_id'] for x in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate challenge IDs")
    if EXPECTED_NOV12_ID not in ids:
        raise ValueError("November 12 challenge is absent from evaluated cohort")
    latest = rows[ids.index(EXPECTED_NOV12_ID)]
    if (latest['settlement_date'],latest['official_label'],latest['predicted_label']) != ('2026-11-12','neutral','bullish'):
        raise ValueError("Nov 12 official settlement or predicted direction mismatch")

    pred_list = []; actual_list = []; p_list = []
    for r in rows:
        a = r['official_label']; pred = r['predicted_label']
        if a not in CLASSES or pred not in CLASSES:
            raise ValueError("Invalid official/predicted class")
        p = [float(r['p_'+c]) for c in CLASSES]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or abs(sum(p)-1) > 1e-8:
            raise ValueError("Invalid full-precision probabilities: " + r['challenge_id'])
        expected_pred = CLASSES[max(range(3),key=lambda i:p[i])]
        if pred != expected_pred:
            raise ValueError("Predicted label/probability class-order mismatch: " + r['challenge_id'])
        actual_list.append(a);pred_list.append(pred);p_list.append(p)
    metrics = _score(p_list,actual_list)
    old = payload.get('notebook_metrics') or {}
    for k in ('n','accuracy','brier','log_loss','avg_confidence'):
        if k not in old or not math.isclose(float(old[k]),float(metrics[k]),rel_tol=1e-10,abs_tol=1e-11):
            raise ValueError("Independent metric disagrees with Notebook 12: " + k)
    matrix = {a:{p:0 for p in CLASSES} for a in CLASSES}
    for a,p in zip(actual_list,pred_list):matrix[a][p]+=1
    classwise=[]
    for c in CLASSES:
        tp=matrix[c][c]; support=sum(matrix[c].values());chosen=sum(matrix[a][c] for a in CLASSES)
        precision=tp/chosen if chosen else None
        recall=tp/support if support else None
        f1=2*tp/(chosen+support) if (chosen+support) else None
        classwise.append(dict(class_label=c, support=support,predicted_count=chosen,tp=tp,
                              precision=precision,recall=recall,f1=f1))
    recalls=[c['recall'] for c in classwise if c['recall'] is not None]
    f1s=[c['f1'] for c in classwise if c['support']>0]
    balanced=sum(recalls)/len(recalls) if recalls else None
    macrof1=sum(f1s)/len(f1s) if f1s else None
    uniform=_score([[1/3]*3 for _ in rows],actual_list)
    prior=payload.get('training_only_prior')
    training_prior=None
    if prior is not None:
        if len(prior)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in prior) or abs(sum(prior)-1)>1e-8:
            raise ValueError("Invalid frozen training-only prior vector")
        training_prior=_score([prior for _ in rows],actual_list)
    historical=[r for r in rows if r['challenge_id'] != EXPECTED_NOV12_ID]
    previous=_score([[float(r['p_'+c]) for c in CLASSES] for r in historical], [r['official_label'] for r in historical])
    if len(historical)!=17 or not math.isclose(previous['brier'],0.610223,abs_tol=0.001):
        raise ValueError("Reconstructed 17-settlement Brier does not agree with prior checkpoint; review source cohort")
    return dict(
       report_type='Phase A frozen M1, original Notebook12 first-30-accepted cohort, interim descriptive checkpoint',
       disclosure='NOT complete Phase A, NOT independent server UTC certified, no inference of skill or profitability',
       settled_count=len(rows),
       accepted_evidenced_n=payload.get('accepted_evidenced_n'),
       official_rows_available_n=payload.get('official_settlement_rows'),
       peer_rows_available_n=payload.get('peer_rows'),
       strict_native_settled_n=payload.get('strict_native_settled_n'),
       excluded_settled_first30_n=payload.get('excluded_settled_first30_n'),
       notebook_study_status=payload.get('notebook_study_status'),
       classes=list(CLASSES),metrics=metrics,confusion_matrix_actual_by_predicted=matrix,
       classwise=classwise,balanced_accuracy=balanced,macro_f1=macrof1,
       baselines={'uniform_brier':uniform['brier'],
                  'uniform_log_loss':uniform['log_loss'],
                  'always_constant_class_accuracy':{c:actual_list.count(c)/len(rows) for c in CLASSES},
                  'training_prior_scores': training_prior,
                  'training_prior_probabilities_bearish_neutral_bullish':prior},
       prior_17_settlement_checkpoint=previous,
       change_from_17={'accuracy':metrics['accuracy']-previous['accuracy'],
                       'brier':metrics['brier']-previous['brier'],
                       'log_loss':metrics['log_loss']-previous['log_loss']},
       full_precision_rows=rows,
    )


def format_summary(report:dict)->str:
    m=report['metrics']; b=report['baselines']; out=[]
    out.extend([
       '# Phase A — 18-settlement descriptive checkpoint',
       '',
       '**Population:** Frozen M1 original first-30-accepted cohort, currently 18 accepted-and-officially-settled ES challenges. NOT the independently certified eligible-opportunity census.',
       '',
       f"- Evaluated settlements: **{m['n']}**",
       f"- Correct classification accuracy: **{m['accuracy']:.6%}**",
       f"- Unscaled multiclass Brier loss (full-precision): **{m['brier']:.12f}**",
       f"- Multiclass log loss (full-precision): **{m['log_loss']:.12f}**",
       f"- Balanced accuracy: **{report['balanced_accuracy']:.6%}**",
       f"- Macro-F1: **{report['macro_f1']:.6f}**",
       f"- Average max-class confidence: **{m['avg_confidence']:.6%}**",
       f"- Uniform probability Brier: **{b['uniform_brier']:.12f}**",
       f"- Training-prior Brier: **{b['training_prior_scores']['brier']:.12f}**" if b['training_prior_scores'] is not None else '- Training prior unavailable: not estimated from evaluation outcomes',
       f"- Prior 17-settlement reconstructed Brier: **{report['prior_17_settlement_checkpoint']['brier']:.12f}**",
       f"- Brier change vs 17 settlements: **{report['change_from_17']['brier']:+.12f}**",
       '',
       '## Confusion matrix (actual rows × predicted columns)',
       '',
       '| Actual / Predicted | Bearish | Neutral | Bullish |',
       '|---|---:|---:|---:|',
    ])
    for a in CLASSES:
        d=report['confusion_matrix_actual_by_predicted'][a]
        out.append(f"| {a.title()} | {d['bearish']} | {d['neutral']} | {d['bullish']} |")
    out+=['','## Classwise scores','','| Class | Actual n | Predicted n | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|']
    for c in report['classwise']:
        def fmt(v):return 'undefined' if v is None else f'{v:.6f}'
        out.append(f"| {c['class_label']} | {c['support']} | {c['predicted_count']} | {fmt(c['precision'])} | {fmt(c['recall'])} | {fmt(c['f1'])} |")
    out.extend(['','## Operational and scientific limits','',
        f"- Official rows available: {report['official_rows_available_n']}; peer rows available: {report['peer_rows_available_n']}.",
        f"- Strict independent-UTC-certified native settled n: {report['strict_native_settled_n']}.",
        f"- Notebook status: {report['notebook_study_status']}.",
        '- Arena outcomes are authoritative. Original frozen probabilities are not regenerated.',
        '- 18 observations do not establish significant forecasting superiority or profitability.',
        '- Do not publish the executed private notebook or detailed full-precision rows without review.',
        ''])
    return '\n'.join(out)


def execute_notebook(path: Path, root: Path, timeout: int=180, kernel: str="python3"):
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as e:
        raise RuntimeError("Missing notebook dependencies. Install nbclient, nbformat, ipykernel in your active .venv") from e
    nb=nbformat.read(path,as_version=4)
    source='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    if not re.search(r'WRITE_MISSING_TEMPLATES\s*=\s*False',source):
        raise RuntimeError("Notebook write-safety default absent: STOP")
    if not re.search(r'FROZEN_MODEL_SHA256\s*=\s*"5e8a2a9f3ba9a310b340aa09c3ddc823aab2fe9aed728ac87965c2e61a5ae68d"',source):
        raise RuntimeError("Unexpected frozen model identity in Notebook 12")
    nb.cells.append(nbformat.v4.new_code_cell(EXPORT_CELL))
    client=NotebookClient(nb,timeout=timeout,kernel_name=kernel,allow_errors=False,
                          resources={'metadata':{'path':str(root)}})
    executed=client.execute()
    outputs=executed.cells[-1].get('outputs',[])
    found=[]
    for output in outputs:
        if output.output_type=='stream':
            for line in output.text.splitlines():
                if line.startswith(MARKER): found.append(json.loads(line[len(MARKER):]))
    if len(found)!=1: raise RuntimeError("Expected exactly one complete evaluation export payload")
    return executed,found[0]


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='.',help='Market-Mood-Forecasting Git repository root')
    ap.add_argument('--notebook',default=DEFAULT_NOTEBOOK,help='Source public Notebook 12, relative to root')
    ap.add_argument('--out-dir',required=True,help='NEW private directory OUTSIDE repository')
    ap.add_argument('--expected-n',type=int,default=18,help='Refuse to report any other sample count (default 18)')
    ap.add_argument('--timeout',type=int,default=180,help='Per-cell execution timeout, seconds')
    ap.add_argument('--kernel',default='python3',help='Jupyter kernel to use for execution (default python3)')
    args=ap.parse_args(argv)
    root=Path(args.root).resolve()
    if not (root/'es_arena_daily.py').is_file(): raise FileNotFoundError('Project root/es_arena_daily.py missing')
    nb_path=(root/args.notebook).resolve()
    if root not in nb_path.parents: raise ValueError('Notebook must be inside project repository')
    if not nb_path.is_file(): raise FileNotFoundError('Notebook not found: '+str(nb_path))
    out=_safe_outdir(Path(args.out_dir),root)
    print('SOURCE NOTEBOOK SHA256:',sha256_file(nb_path))
    print('EXECUTING READ-ONLY NOTEBOOK. Output withheld unless validations pass.')
    executed,payload=execute_notebook(nb_path,root,args.timeout,args.kernel)
    report=compute_report(payload,args.expected_n)
    report['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    report['notebook_source_sha256']=sha256_file(nb_path)
    report['official_csv_sha256']=sha256_file(root/'data/processed/es_arena_research_v0_1/arena_official_settlements.csv')
    report['peer_csv_sha256']=sha256_file(root/'data/processed/es_arena_research_v0_1/arena_peer_forecasts.csv')
    import nbformat
    # Output folder is intentionally made only AFTER successful notebook execution and validation.
    out.mkdir()
    paths={
       'json':out/'phase_a_18_settlement_checkpoint_private.json',
       'markdown':out/'phase_a_18_settlement_summary_private.md',
       'notebook':out/'notebook12_executed_private.ipynb',
    }
    with paths['json'].open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,allow_nan=False)
    paths['markdown'].write_text(format_summary(report),encoding='utf-8')
    nbformat.write(executed,paths['notebook'])
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for p in paths.values(): f.write(sha256_file(p)+'  '+p.name+'\n')
    print('SUCCESS: New private checkpoint created:',out)
    print(f"SETTLED: {report['settled_count']}; ACCURACY: {report['metrics']['accuracy']:.6%}; FULL PRECISION BRIER: {report['metrics']['brier']:.12f}")
    print('SOURCE EVIDENCE FILES CHANGED BY SCRIPT: 0 (notebook safety default also verified)')
    print('PRIVATE REPORTS: JSON, Markdown, executed notebook, checksum list. Do not add these to Git.')
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as e:
        print(f'STOP — {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(2)
