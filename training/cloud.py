"""Immutable GitHub Release checkpoint bundles; local files are backups."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

REPO = 'GAON213193/RL-Maze-Explorer'
BRANCH = 'codex/robot-environment-foundation'
PREFIX = 'ppo-run-'


def gh(*args):
    if not shutil.which('gh'):
        raise RuntimeError('Install GitHub CLI from https://cli.github.com then run gh auth login')
    try:
        result = subprocess.run(['gh', *args], capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=600)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError('GitHub CLI could not complete the request: ' + str(exc)) from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'GitHub CLI failed')
    return result.stdout


def check_access():
    gh('auth', 'status', '--hostname', 'github.com')
    repo = json.loads(gh('api', f'repos/{REPO}'))
    if not repo.get('permissions', {}).get('push'):
        raise RuntimeError('GitHub account needs write access to ' + REPO)


def publish(bundle, tag):
    # An existing uploaded asset is immutable. Never overwrite another checkpoint.
    releases = json.loads(gh('api', f'repos/{REPO}/releases', '--paginate', '--slurp'))
    release = next((r for page in releases for r in page if r['tag_name'] == tag), None)
    if release is None:
        gh('release', 'create', tag, str(bundle), '--repo', REPO, '--target', BRANCH,
           '--title', tag, '--notes', 'PPO checkpoints, configuration and training metrics.',
           '--prerelease', '--latest=false')
    elif not any(a['name'] == bundle.name for a in release['assets']):
        gh('release', 'upload', tag, str(bundle), '--repo', REPO)
    bundle.with_suffix('.uploaded').write_text(tag, encoding='utf-8')


def sync_pending(root=Path('runs')):
    for bundle in sorted(root.glob('*/cloud/checkpoint-*.zip')):
        if not bundle.with_suffix('.uploaded').exists():
            tag = json.loads((bundle.parent.parent / 'config.json').read_text())['cloud_tag']
            publish(bundle, tag)
            print('Uploaded backup:', bundle, flush=True)


def extract_bundle(bundle, destination):
    with ZipFile(bundle) as z:
        names = set(z.namelist())
        allowed = {'model.zip', 'config.json', 'metrics.json', 'episodes.csv'}
        if not {'model.zip', 'config.json'} <= names or not names <= allowed or z.testzip():
            raise ValueError('Invalid checkpoint bundle')
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            (destination / name).write_bytes(z.read(name))


def download_latest(prefix=PREFIX):
    pages = json.loads(gh('api', f'repos/{REPO}/releases', '--paginate', '--slurp'))
    candidates = [(a['created_at'], r['created_at'], r['tag_name'], a['name'])
                  for page in pages for r in page
                  if r['tag_name'].startswith(prefix) and not r['draft']
                  for a in r['assets'] if re.fullmatch(r'checkpoint-\d+\.zip', a['name'])]
    if not candidates:
        return None
    _, _, tag, name = max(candidates)
    # Fresh directory avoids using a stale local file after a failed download.
    folder = Path(tempfile.mkdtemp(prefix='ppo-download-'))
    try:
        gh('release', 'download', tag, '--repo', REPO, '--pattern', name, '--dir', str(folder))
        destination = Path('runs') / 'downloads' / tag / name.removesuffix('.zip')
        extract_bundle(folder / name, destination)
        print('Downloaded GitHub checkpoint:', tag, name, flush=True)
        return destination
    finally:
        shutil.rmtree(folder)


def make_bundle(output, step):
    folder = output / 'cloud'
    folder.mkdir(exist_ok=True)
    path = folder / f'checkpoint-{step:012d}.zip'
    if path.exists():
        return path
    temp = path.with_suffix('.tmp')
    with ZipFile(temp, 'w', compression=ZIP_DEFLATED) as z:
        for name in ('model.zip', 'config.json', 'metrics.json', 'episodes.csv'):
            if (output / name).exists():
                z.write(output / name, name)
    temp.replace(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['sync'])
    parser.parse_args()
    check_access()
    sync_pending()
    print('All pending backups uploaded.')
