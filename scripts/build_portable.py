"""Build a Windows x64 ZIP using official embedded Python and GitHub CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination):
    request = urllib.request.Request(url, headers={'User-Agent': 'RL-Maze-Explorer-portable-builder'})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open('wb') as output:
        shutil.copyfileobj(response, output)


def build(output, zig):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'downloads'
    cache.mkdir(exist_ok=True)
    folder = output / 'RL-Maze-Explorer-Web'
    if folder.exists():
        raise ValueError('Use a fresh output directory to avoid mixing runtime versions.')
    folder.mkdir()
    runtime = folder / 'runtime'
    runtime.mkdir()
    embedded = cache / 'python-3.12.10-embed-amd64.zip'
    download('https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip', embedded)
    with ZipFile(embedded) as archive:
        archive.extractall(runtime)
    (runtime / 'python312._pth').write_text('python312.zip\n.\nLib/site-packages\n..\nimport site\n', encoding='utf-8')
    site = runtime / 'Lib/site-packages'
    site.mkdir(parents=True)
    with urllib.request.urlopen('https://pypi.org/pypi/pip/25.3/json', timeout=30) as response:
        metadata = json.load(response)
    wheel = next(file for file in metadata['urls'] if file['filename'].endswith('.whl'))
    pip_file = cache / wheel['filename']
    download(wheel['url'], pip_file)
    if hashlib.sha256(pip_file.read_bytes()).hexdigest() != wheel['digests']['sha256']:
        raise ValueError('pip wheel checksum mismatch')
    with ZipFile(pip_file) as archive:
        archive.extractall(site)
    gh_file = cache / 'gh_2.83.0_windows_amd64.zip'
    download('https://github.com/cli/cli/releases/download/v2.83.0/gh_2.83.0_windows_amd64.zip', gh_file)
    with ZipFile(gh_file) as archive:
        archive.extractall(folder / '.tools/gh')
    if not (folder / '.tools/gh/bin/gh.exe').is_file():
        raise ValueError('GitHub CLI binary is missing')
    for name in ('training', 'environment'):
        shutil.copytree(ROOT / name, folder / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'requirements-rl.txt', folder)
    shutil.copy2(ROOT / 'requirements-desktop.txt', folder)
    shutil.copy2(ROOT / 'docs/DESKTOP.md', folder / '사용방법.txt')
    subprocess.run([str(zig), 'cc', '-target', 'x86_64-windows-gnu', '-Os', '-s', '-municode',
                    '-Wl,--subsystem,windows', str(ROOT / 'scripts/windows_launcher.c'),
                    '-o', str(folder / 'RL-Maze-Explorer.exe')], check=True)
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (embedded, pip_file, gh_file)}
    (folder / 'runtime-sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    target = output / 'RL-Maze-Explorer-Web-Windows.zip'
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print(target.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zig', type=Path, required=True)
    options = parser.parse_args()
    build(options.output.resolve(), options.zig.resolve())
