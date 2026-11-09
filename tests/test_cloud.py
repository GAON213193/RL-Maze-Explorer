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
            with patch('training.cloud.gh', side_effect=['[[]]', '']) as mocked, patch('training.cloud.index_checkpoint'):
                sync_pending(root)
                self.assertIn('create',mocked.call_args.args)
            self.assertTrue(bundle.with_suffix('.uploaded').exists())

    def test_existing_asset_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle=self.fixture(Path(folder))
            result=[[{'tag_name':'ppo-run-test','assets':[{'name':bundle.name}]}]]
            with patch('training.cloud.gh', return_value=json.dumps(result)) as mocked, patch('training.cloud.index_checkpoint'):
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

    def test_v3_ignores_v2(self):
        releases=[[{'tag_name':'ppo-v2-old','draft':False,'created_at':'2026-10-25',
                    'assets':[{'name':'checkpoint-200192.zip','created_at':'2026-10-25T03:00'}]}]]
        with patch('training.cloud.gh', return_value=json.dumps(releases)) as mocked:
            self.assertIsNone(download_latest(prefix='ppo-v3-'))
            self.assertEqual(mocked.call_count,1)

    def test_network_failure_is_not_new_training(self):
        with patch('training.cloud.gh', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError): download_latest()

    def test_index_failure_keeps_release_and_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); bundle = self.fixture(root)
            release = [[{'tag_name': 'ppo-run-test', 'assets': [{'name': bundle.name}]}]]
            with patch('training.cloud.gh', return_value=json.dumps(release)), patch(
                    'training.cloud.index_checkpoint', side_effect=RuntimeError('offline')):
                with self.assertRaises(RuntimeError): publish(bundle, 'ppo-run-test')
            self.assertTrue(bundle.with_suffix('.uploaded').exists())
            self.assertFalse(bundle.with_suffix('.indexed').exists())
            with patch('training.cloud.gh', return_value=json.dumps(release)) as remote, patch(
                    'training.cloud.index_checkpoint') as index:
                sync_pending(root)
            index.assert_called_once_with(bundle, 'ppo-run-test')
            self.assertEqual(remote.call_count, 1)
            self.assertTrue(bundle.with_suffix('.indexed').exists())

    def test_model_branch_preserves_existing_files_and_checksums(self):
        from training.cloud import index_checkpoint
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            bundle = self.fixture(Path(folder))
            responses = [json.dumps({'object': {'sha': 'head'}}),
                         json.dumps({'tree': {'sha': 'base-tree'}}), json.dumps({'tree': []})]
            payloads = []
            def fake_api(endpoint, payload, method='POST'):
                payloads.append((endpoint, payload, method))
                return {'sha': 'new-object'}
            with patch('training.cloud.gh', side_effect=responses), patch('training.cloud.api_json', side_effect=fake_api):
                index_checkpoint(bundle, 'ppo-run-test')
            tree = next(p for e,p,m in payloads if e.endswith('/git/trees'))
            self.assertEqual(tree['base_tree'], 'base-tree')
            manifest = json.loads(next(e['content'] for e in tree['tree'] if e['path']=='manifest.json'))
            record = manifest['runs'][0]
            self.assertEqual(record['checkpoint_steps'], 128)
            self.assertEqual(record['files_sha256']['model.zip'], hashlib.sha256(b'model fixture').hexdigest())
            self.assertTrue(all(e['path'].startswith('checkpoints/') or e['path'] in ('latest.json','manifest.json') for e in tree['tree']))
            self.assertEqual(payloads[-1][2], 'PATCH')
            self.assertFalse(payloads[-1][1]['force'])

    def test_old_checkpoint_does_not_replace_newer_model(self):
        from training.cloud import index_checkpoint
        import base64
        with tempfile.TemporaryDirectory() as folder:
            bundle = self.fixture(Path(folder))
            manifest = {'runs': [{'run': 'ppo-run-test', 'checkpoint_steps': 256}]}
            responses = [{'object': {'sha': 'head'}}, {'tree': {'sha': 'base'}},
                         {'tree': [{'path': 'manifest.json', 'sha': 'manifest'}]},
                         {'content': base64.b64encode(json.dumps(manifest).encode()).decode()}]
            with patch('training.cloud.gh', side_effect=[json.dumps(r) for r in responses]), patch('training.cloud.api_json') as write:
                index_checkpoint(bundle, 'ppo-run-test')
            write.assert_not_called()

if __name__=='__main__': unittest.main()

