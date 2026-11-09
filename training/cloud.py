"""Immutable GitHub Release checkpoint bundles; local files are backups."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

REPO = 'GAON213193/RL-Maze-Explorer'
BRANCH = 'codex/browser-training-launcher'
MODEL_BRANCH = 'models'
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
    if re.fullmatch(r'checkpoint-\d+\.zip', bundle.name):
        index_checkpoint(bundle, tag)
        bundle.with_suffix('.indexed').write_text(tag, encoding='utf-8')


def api_json(endpoint, payload, method='POST'):
    with tempfile.TemporaryDirectory(prefix='ppo-api-') as folder:
        request = Path(folder) / 'request.json'
        request.write_text(json.dumps(payload), encoding='utf-8')
        return json.loads(gh('api', endpoint, '--method', method, '--input', str(request)))


def index_checkpoint(bundle, tag):
    """Atomically update the model-only branch without moving code or Release tags."""
    if not re.fullmatch(r'ppo-(?:v\d+|run)-[A-Za-z0-9-]+', tag):
        raise ValueError('Invalid model run tag')
    step = int(bundle.stem.removeprefix('checkpoint-'))
    with tempfile.TemporaryDirectory(prefix='ppo-index-') as folder:
        restored = Path(folder) / 'checkpoint'
        extract_bundle(bundle, restored)
        files = {p.name: p.read_bytes() for p in restored.iterdir()}
    endpoint = f'repos/{REPO}'
    head = json.loads(gh('api', f'{endpoint}/git/ref/heads/{MODEL_BRANCH}'))['object']['sha']
    commit = json.loads(gh('api', f'{endpoint}/git/commits/{head}'))
    tree = json.loads(gh('api', f'{endpoint}/git/trees/{commit["tree"]["sha"]}'))
    entries = {entry['path']: entry for entry in tree['tree']}

    def read_json(name, default):
        if name not in entries:
            return default
        blob = json.loads(gh('api', f'{endpoint}/git/blobs/{entries[name]["sha"]}'))
        return json.loads(base64.b64decode(blob['content']))

    manifest = read_json('manifest.json', {'runs': []})
    previous = next((r for r in manifest['runs'] if r['run'] == tag), None)
    if previous and previous['checkpoint_steps'] >= step:
        return
    record = dict(run=tag, checkpoint_steps=step,
                  source_release=f'https://github.com/{REPO}/releases/tag/{tag}',
                  files_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    manifest['runs'] = [r for r in manifest['runs'] if r['run'] != tag] + [record]
    changes = []
    for name, data in files.items():
        blob = api_json(f'{endpoint}/git/blobs',
                        {'content': base64.b64encode(data).decode('ascii'), 'encoding': 'base64'})
        changes.append(dict(path=f'checkpoints/{tag}/{name}', mode='100644', type='blob', sha=blob['sha']))
    changes.append(dict(path='manifest.json', mode='100644', type='blob',
                        content=json.dumps(manifest, indent=2)))
    latest = read_json('latest.json', {})
    # Timestamp is embedded in the run tag, independent of the profile prefix.
    def run_time(run):
        match = re.search(r'(\d{8}-\d{6})', run)
        return match.group(1) if match else ''
    if run_time(tag) >= run_time(latest.get('run', '')):
        changes.append(dict(path='latest.json', mode='100644', type='blob', content=json.dumps(
            dict(run=tag, total_steps=step, path=f'checkpoints/{tag}',
                 source_release=record['source_release']), indent=2)))
    updated_tree = api_json(f'{endpoint}/git/trees', {'base_tree': commit['tree']['sha'], 'tree': changes})
    updated_commit = api_json(f'{endpoint}/git/commits',
        {'message': f'Index {tag} at {step} steps', 'tree': updated_tree['sha'], 'parents': [head]})
    # GitHub rejects a concurrent non-fast-forward update; sync retries from the new head.
    api_json(f'{endpoint}/git/refs/heads/{MODEL_BRANCH}',
             {'sha': updated_commit['sha'], 'force': False}, method='PATCH')


def sync_pending(root=Path('runs')):
    for bundle in sorted(root.glob('*/cloud/checkpoint-*.zip')):
        if not bundle.with_suffix('.uploaded').exists() or not bundle.with_suffix('.indexed').exists():
            tag = json.loads((bundle.parent.parent / 'config.json').read_text())['cloud_tag']
            publish(bundle, tag)
            print('Uploaded backup:', bundle, flush=True)
    for pending in sorted(root.rglob('*.evaluation-pending.json')):
        record=json.loads(pending.read_text(encoding='utf-8'))
        bundle=pending.parent / record['bundle']
        if not bundle.with_suffix('.uploaded').exists():
            publish(bundle,record['tag'])
            print('Uploaded evaluation:',bundle,flush=True)



def extract_bundle(bundle, destination):
    with ZipFile(bundle) as z:
        names = set(z.namelist())
        allowed = {'model.zip', 'config.json', 'metrics.json', 'episodes.csv', 'training_summary.txt', 'training_history.csv'}
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
        for name in ('model.zip', 'config.json', 'metrics.json', 'episodes.csv', 'training_summary.txt', 'training_history.csv'):
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

