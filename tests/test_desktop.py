import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from training.desktop import Launcher, make_server


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = Launcher(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_request_auth_and_cross_origin(self):
        server = make_server(self.app)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        url = 'http://' + server.address
        try:
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url + '/api/status')
            self.assertEqual(error.exception.code, 403)
            request = urllib.request.Request(url + '/api/status', headers={'X-Launcher-Token': self.app.token})
            with urllib.request.urlopen(request) as response:
                self.assertFalse(json.load(response)['running'])
            request.add_header('Origin', 'https://example.com')
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)
            request = urllib.request.Request(url + '/api/status', headers={'Host': 'example.com', 'X-Launcher-Token': self.app.token})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_jobs_validate_resume_and_save_stop_request(self):
        request = dict(job='resume', size=15, steps=128, lesson=0.3, cloud=False, resume='../../outside')
        with patch.object(self.app, 'ready', return_value={'ready': True}):
            with self.assertRaises(ValueError):
                self.app.start(request)
            folder = self.app.root / 'runs/example'
            folder.mkdir(parents=True)
            (folder / 'model.zip').touch()
            (folder / 'config.json').write_text(json.dumps({'profile': 'range-charge-v3', 'environment': {'size': 7}}))
            request['resume'] = 'runs/example'
            with patch('training.desktop.threading.Thread') as thread:
                self.app.start(request)
                command = thread.call_args.kwargs['args'][0]
                self.assertIn('runs/example', command)
                self.assertEqual(command[command.index('--size') + 1], '15')
                self.assertIn('--stop-file', command)
                with self.assertRaises(ValueError):
                    self.app.start(request)
                self.app.stop()
                self.assertTrue(self.app.stop_file.exists())
                self.assertTrue(self.app.status()['stopping'])

    def test_process_errors_release_job_and_keep_log(self):
        self.app.running = True
        self.app._run([str(self.app.root / 'missing-program')])
        self.assertFalse(self.app.running)
        self.assertEqual(self.app.exit_code, 1)
        self.assertTrue(self.app.lines)


if __name__ == '__main__':
    unittest.main()
