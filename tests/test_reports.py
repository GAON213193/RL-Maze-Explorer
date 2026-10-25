import csv
import json
import tempfile
import unittest
from pathlib import Path
from training.reports import training_report, rate, evaluation_report
from training.cloud import make_bundle,extract_bundle

class ReportTests(unittest.TestCase):
    def test_no_episodes_is_not_zero_success(self):
        self.assertEqual(rate(0,0),'집계된 판 없음')

    def test_training_history_and_scenario_denominators(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'config.json').write_text(json.dumps({'environment':{'size':64,'charge_turns':2}}))
            metrics=dict(total_steps=128,episodes=10,successes=3,battery_depletions=6,stalls=1,
                         recent_100_success_rate=.3,success_rate=.3,mean_turns=12,
                         scenarios={'navigation':dict(episodes=8,successes=2,episodes_with_charge=2,success_after_charge=1)})
            training_report(root,metrics);training_report(root,metrics)
            text=(root/'training_summary.txt').read_text(encoding='utf-8-sig')
            self.assertIn('25.0% (2/8)',text)
            self.assertIn('50.0% (1/2)',text)
            with (root/'training_history.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(len(list(csv.DictReader(f))),1)
            metrics['total_steps']=256
            training_report(root,metrics)
            with (root/'training_history.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(len(list(csv.DictReader(f))),2)
            (root/'model.zip').write_bytes(b'test')
            bundle=make_bundle(root,256)
            extract_bundle(bundle,root/'restore')
            self.assertEqual((root/'restore/training_summary.txt').read_bytes(),(root/'training_summary.txt').read_bytes())

    def test_evaluation_reports_modes_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            row=dict(episodes=4,outcomes={'success':2},charging={'episodes_with_charge':1,'success_after_charge':1},
                     deterministic=True,size=64,seed=12,mean_turns=7,scenarios={'navigation':4})
            meta=dict(total_steps=128,model_sha256='test-hash',profile='range-charge-v3',scenario='navigation')
            files=evaluation_report(Path(d)/'evaluation.json',[row],meta)
            self.assertTrue(all(p.exists() for p in files))
            text=(Path(d)/'evaluation.txt').read_text(encoding='utf-8-sig')
            self.assertIn('50.0% (2/4)',text)
            self.assertIn('100.0% (1/1)',text)
            self.assertIn('test-hash',text)

if __name__=='__main__':unittest.main()
