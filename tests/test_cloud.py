import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from training.cloud import make_bundle, extract_bundle, publish, sync_pending, download_latest

class CloudTests(unittest.TestCase):
    def fixture(self, root):
        output = root/'run'
        output.mkdir()
        (output/'model.zip').write_bytes(b'model fixture')
        (output/'config.json').write_text(json.dumps({'cloud_tag':'ppo-run-test'}))
        return make_bundle(output, 128)

    def test_bundle_roundtrip_and_path_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); bundle=self.fixture(root)
            extract_bundle(bundle, root/'restore')
            self.assertEqual((root/'restore/model.zip').read_bytes(), b'model fixture')
            bad=root/'bad.zip'
            with ZipFile(bad,'w') as z:
                z.writestr('../escape','bad'); z.writestr('model.zip','x'); z.writestr('config.json','{}')
            with self.assertRaises(ValueError): extract_bundle(bad, root/'bad')

    def test_failed_upload_retained_and_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); bundle=self.fixture(root)
            with patch('training.cloud.gh', side_effect=RuntimeError('offline')):
                with self.assertRaises(RuntimeError): publish(bundle, 'ppo-run-test')
            self.assertTrue(bundle.exists())
            self.assertFalse(bundle.with_suffix('.uploaded').exists())
            with patch('training.cloud.gh', side_effect=['[[]]', '']) as mocked:
                sync_pending(root)
                self.assertIn('create',mocked.call_args.args)
            self.assertTrue(bundle.with_suffix('.uploaded').exists())

    def test_existing_asset_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle=self.fixture(Path(folder))
            result=[[{'tag_name':'ppo-run-test','assets':[{'name':bundle.name}]}]]
            with patch('training.cloud.gh', return_value=json.dumps(result)) as mocked:
                publish(bundle,'ppo-run-test')
                self.assertEqual(mocked.call_count,1)

    def test_latest_download(self):
        releases=[[{'tag_name':'ppo-run-new','draft':False,'created_at':'2026-10-25',
                    'assets':[{'name':'checkpoint-000128.zip','created_at':'2026-10-25T01:00'},
                              {'name':'checkpoint-000256.zip','created_at':'2026-10-25T02:00'}]}]]
        def fake(*args):
            if args[0]=='api': return json.dumps(releases)
            self.assertIn('checkpoint-000256.zip',args)
            return ''
        with patch('training.cloud.gh',side_effect=fake), patch('training.cloud.extract_bundle') as extract:
            result=download_latest()
            self.assertEqual(result.name,'checkpoint-000256')
            extract.assert_called_once()

    def test_v2_ignores_archived_v1(self):
        releases=[[{'tag_name':'ppo-run-old','draft':False,'created_at':'2026-10-25',
                    'assets':[{'name':'checkpoint-100096.zip','created_at':'2026-10-25T03:00'}]}]]
        with patch('training.cloud.gh', return_value=json.dumps(releases)) as mocked:
            self.assertIsNone(download_latest(prefix='ppo-v2-'))
            self.assertEqual(mocked.call_count,1)

    def test_network_failure_is_not_new_training(self):
        with patch('training.cloud.gh', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError): download_latest()

if __name__=='__main__': unittest.main()
