"""Loopback-only browser controls for the portable Windows distribution."""
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from urllib.parse import urlsplit, parse_qs
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


class Launcher:
    def __init__(self, root=ROOT):
        self.root = root.resolve()
        self.python = (self.root / 'runtime/python.exe') if os.name == 'nt' else Path(sys.executable)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.running = False
        self.label = ''
        self.exit_code = None
        self.lines = deque(maxlen=500)
        self.stop_file = None
        self.stopping = False
        self.closing = False
        self.env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        self.env['PATH'] = str(self.root / '.tools/gh/bin') + os.pathsep + self.env.get('PATH', '')

    def runs(self):
        result = []
        for config in (self.root / 'runs').rglob('config.json'):
            folder = config.parent
            if not (folder / 'model.zip').is_file():
                continue
            try:
                data = json.loads(config.read_text(encoding='utf-8-sig'))
                if data.get('profile') != 'range-charge-v3':
                    continue
                identifier = folder.relative_to(self.root).as_posix()
                result.append({'id': identifier, 'label': f"{identifier} ({data['environment']['size']}×{data['environment']['size']})", 'mtime': (folder / 'model.zip').stat().st_mtime})
            except (OSError, ValueError, KeyError):
                continue
        return sorted(result, key=lambda r: r['mtime'], reverse=True)

    def status(self):
        with self.lock:
            result = dict(running=self.running, label=self.label, exit_code=self.exit_code,
                          log='\n'.join(self.lines), can_stop=self.running and self.stop_file is not None,
                          stopping=self.stopping)
        result['runs'] = self.runs()
        return result

    def ready(self):
        try:
            result = subprocess.run([str(self.python), '-c', 'import torch, stable_baselines3, gymnasium, numpy'],
                                    cwd=self.root, env=self.env, capture_output=True, timeout=45,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            detail = result.stderr.decode('utf-8', errors='replace')[-1200:]
            return {'ready': result.returncode == 0, 'detail': '「학습 환경 준비」를 눌러 패키지를 설치하세요.\n' + detail}
        except (OSError, subprocess.TimeoutExpired):
            return {'ready': False, 'detail': 'Python 실행 상태를 확인하지 못했습니다. 압축을 모두 풀었는지 확인하세요.'}

    def start(self, request):
        job = request.get('job')
        stop_file = None
        if job == 'setup':
            command = [str(self.python), '-m', 'pip', 'install', '--disable-pip-version-check',
                       '--only-binary=:all:', '--no-deps', '--no-warn-script-location',
                       '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                       '--target', str(self.root / 'runtime/Lib/site-packages'),
                       '--upgrade', '-r', str(self.root / 'requirements-desktop.txt')]
            label = '학습 환경 준비'
        elif job == 'login':
            command = [str(self.root / '.tools/gh/bin/gh.exe'), 'auth', 'login',
                       '--hostname', 'github.com', '--git-protocol', 'https', '--web']
            label = 'GitHub 로그인'
        elif job in ('new', 'resume'):
            size, steps, lesson = request.get('size'), request.get('steps'), request.get('lesson')
            if type(size) is not int or size not in (7, 15, 31, 64):
                raise ValueError('미로 크기를 선택하세요.')
            if type(steps) is not int or not 128 <= steps <= 10_000_000:
                raise ValueError('학습 스텝은 128~10,000,000 사이로 입력하세요.')
            if type(lesson) not in (int, float) or lesson not in (0.3, 0.5, 0.8, 1):
                raise ValueError('귀환 연습 비율을 선택하세요.')
            if not self.ready()['ready']:
                raise ValueError('먼저 학습 환경 준비를 완료하세요.')
            if type(request.get('cloud')) is not bool:
                raise ValueError('GitHub 옵션을 확인하세요.')
            stop_file = self.root / '.launcher' / (secrets.token_hex(12) + '.stop')
            output = Path('runs') / ('web-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            command = [str(self.python), '-u', '-m', 'training.train', '--size', str(size),
                       '--steps', str(steps), '--lesson-probability', str(lesson),
                       '--output', str(output), '--stop-file', str(stop_file)]
            if request['cloud']:
                command += ['--cloud']
                if job == 'resume':
                    command += ['--auto-resume']
            elif job == 'resume':
                identifier = request.get('resume')
                allowed = {run['id'] for run in self.runs()}
                if not isinstance(identifier, str) or identifier not in allowed:
                    raise ValueError('이어 학습할 로컬 모델을 선택하세요.')
                command += ['--resume', identifier]
            label = f'{size}×{size} ' + ('새 학습' if job == 'new' else '이어 학습')
        else:
            raise ValueError('지원하지 않는 작업입니다.')
        with self.lock:
            if self.running or self.closing:
                raise ValueError('현재 작업을 마친 뒤 실행하세요.')
            self.running, self.label, self.exit_code = True, label, None
            self.stop_file, self.stopping = stop_file, False
            self.lines.clear()
            threading.Thread(target=self._run, args=(command,), daemon=True).start()

    def _run(self, command):
        code = 1
        try:
            logs = self.root / '.launcher'
            logs.mkdir(exist_ok=True)
            with (logs / 'latest.log').open('w', encoding='utf-8') as log:
                with subprocess.Popen(command, cwd=self.root, env=self.env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      text=True, encoding='utf-8', errors='replace',
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        with self.lock:
                            self.lines.append(line.rstrip())
                    code = process.wait()
        except Exception as error:
            with self.lock:
                self.lines.append(str(error))
        finally:
            with self.lock:
                if self.stop_file:
                    self.stop_file.unlink(missing_ok=True)
                self.running, self.exit_code = False, code

    def stop(self):
        with self.lock:
            if not self.running or self.stop_file is None:
                raise ValueError('중지할 학습이 없습니다.')
            self.stop_file.parent.mkdir(exist_ok=True)
            self.stop_file.touch()
            self.stopping = True


def make_server(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, data, mime='application/json; charset=utf-8'):
            body = data.encode('utf-8') if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', f"default-src 'self'; script-src 'nonce-{app.token}'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self, page=False):
            if self.headers.get('Host') != self.server.address:
                return False
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.server.address:
                return False
            supplied = parse_qs(urlsplit(self.path).query).get('token', [''])[0] if page else self.headers.get('X-Launcher-Token', '')
            return hmac.compare_digest(supplied.encode('utf-8'), app.token.encode('utf-8'))

        def do_GET(self):
            path = urlsplit(self.path).path
            if not self.authorized(page=path == '/'):
                return self.send(403, {'error': '실행 파일을 다시 켜서 화면을 여세요.'})
            if path == '/':
                html = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
                return self.send(200, html.replace('__NONCE__', app.token), 'text/html; charset=utf-8')
            if path == '/api/status':
                return self.send(200, app.status())
            if path == '/api/ready':
                return self.send(200, app.ready())
            self.send(404, {'error': '없는 페이지입니다.'})

        def do_POST(self):
            if not self.authorized():
                return self.send(403, {'error': '허용되지 않은 요청입니다.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('잘못된 요청 크기입니다.')
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('잘못된 요청입니다.')
                if self.path == '/api/start':
                    app.start(request)
                elif self.path == '/api/stop':
                    app.stop()
                elif self.path == '/api/shutdown':
                    with app.lock:
                        if app.running:
                            raise ValueError('현재 작업을 마친 뒤 종료하세요.')
                        app.closing = True
                        self.send(200, {'ok': True})
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                        return
                else:
                    return self.send(404, {'error': '없는 작업입니다.'})
                self.send(200, {'ok': True})
            except (ValueError, OSError) as error:
                self.send(400, {'error': str(error)})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.address = f'127.0.0.1:{server.server_port}'
    return server


def main():
    app = Launcher()
    (ROOT / '.launcher').mkdir(exist_ok=True)
    with (ROOT / '.launcher/instance.lock').open('a+b') as lock_file:
        if os.name == 'nt':
            import msvcrt
            lock_file.write(b'0')
            lock_file.flush()
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                shortcut = ROOT / '.launcher/open-browser.url'
                if shortcut.exists():
                    os.startfile(shortcut)
                return
        with make_server(app) as server:
            url = f'http://{server.address}/?token={app.token}'
            (ROOT / '.launcher/open-browser.url').write_text('[InternetShortcut]\nURL=' + url, encoding='utf-8')
            webbrowser.open(url)
            server.serve_forever()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if os.name == 'nt':
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(error), 'RL Maze Explorer', 0x10)
        raise
