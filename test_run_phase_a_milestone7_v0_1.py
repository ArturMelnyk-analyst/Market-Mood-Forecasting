import tempfile
import unittest
from pathlib import Path
from run_phase_a_milestone7_v0_1 import _score, _safe_outdir, compute_report, format_summary, EXPECTED_NOV12_ID, CLASSES


def fixture():
    # Not representative of real results, synthetic only; 17-settlement Brier 0.610223.
    # For a 3-class target, p actual chosen such that Brier is near 0.610223.
    import math
    # for true bullish with other two equal: loss=1.5*(1-p)^2
    p=1-math.sqrt(0.610223/1.5)
    rows=[]
    for i in range(17):
        rows.append(dict(challenge_id=f'synthetic-{i}',settlement_date=f'2026-10-{20+i:02}',
                         official_label='bullish', predicted_label='bullish',
                         p_bearish=(1-p)/2,p_neutral=(1-p)/2,p_bullish=p))
    rows.append(dict(challenge_id=EXPECTED_NOV12_ID,settlement_date='2026-11-12',official_label='neutral',
                     predicted_label='bullish',p_bearish=0.378,p_neutral=0.176,p_bullish=0.446))
    m=_score([[r['p_'+c] for c in CLASSES] for r in rows],[r['official_label'] for r in rows])
    return dict(rows=rows,notebook_metrics=m,strict_native_settled_n=0,accepted_evidenced_n=19,
                official_settlement_rows=18,peer_rows=57,training_only_prior=None,
                notebook_study_status='PRELIMINARY_OR_INCOMPLETE',excluded_settled_first30_n=0)

class CheckpointTests(unittest.TestCase):
    def test_synthetic_valid_18(self):
        r=compute_report(fixture());self.assertEqual(r['settled_count'],18)
        self.assertEqual(sum(sum(v.values()) for v in r['confusion_matrix_actual_by_predicted'].values()),18)
    def test_no_duplicate_ids(self):
        x=fixture();x['rows'][0]['challenge_id']=x['rows'][1]['challenge_id']
        with self.assertRaisesRegex(ValueError,'Duplicate'):compute_report(x)
    def test_wrong_latest_outcome(self):
        x=fixture();x['rows'][-1]['official_label']='bearish'
        with self.assertRaisesRegex(ValueError,'Nov 12'):compute_report(x)
    def test_wrong_count(self):
        x=fixture();x['rows'].pop()
        with self.assertRaisesRegex(ValueError,'Expected 18'):compute_report(x)
    def test_invalid_probabilities(self):
        x=fixture();x['rows'][-1]['p_bullish']=.9
        with self.assertRaisesRegex(ValueError,'probabilities'):compute_report(x)
    def test_top_class_mismatch(self):
        x=fixture();x['rows'][-1]['predicted_label']='bearish'
        with self.assertRaisesRegex(ValueError,'Nov 12'):compute_report(x)
    def test_nonlatest_predicted_class_mismatch(self):
        x=fixture();x["rows"][0]["predicted_label"]="bearish"
        with self.assertRaisesRegex(ValueError,"class-order"):compute_report(x)
    def test_notebook_metric_mismatch(self):
        x=fixture();x['notebook_metrics']['brier']+=.2
        with self.assertRaisesRegex(ValueError,'disagrees'):compute_report(x)
    def test_historical_brier_mismatch(self):
        x=fixture();x['rows'][0]['p_bullish']=.8;x['rows'][0]['p_bearish']=.1;x['rows'][0]['p_neutral']=.1
        x['notebook_metrics']=_score([[r['p_'+c] for c in CLASSES] for r in x['rows']],[r['official_label'] for r in x['rows']])
        with self.assertRaisesRegex(ValueError,'17-settlement Brier'):compute_report(x)
    def test_has_uniform_reference(self):
        r=compute_report(fixture());self.assertAlmostEqual(r['baselines']['uniform_brier'],2/3)
    def test_summary_text(self):
        self.assertIn('Confusion matrix',format_summary(compute_report(fixture())))
    def test_output_inside_repo_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'data/private'
            with self.assertRaisesRegex(ValueError,'OUTSIDE'): _safe_outdir(out,root)
    def test_output_exists_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'repo';root.mkdir()
            out=Path(t)/'existing';out.mkdir()
            with self.assertRaises(FileExistsError):_safe_outdir(out,root)

if __name__=='__main__':unittest.main()
