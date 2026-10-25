"""Run a bounded PPO job; save model and configuration for resuming."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
from .checkpoints import Checkpoints
from .cloud import check_access, sync_pending, download_latest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from .profiles import make_env
from environment.gym_v3 import PROFILE
from dataclasses import replace
from environment.robot import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cloud', action='store_true', help='Save checkpoints to GitHub Releases')
    p.add_argument('--auto-resume', action='store_true', help='Resume latest GitHub checkpoint if present')
    p.add_argument('--save-every', type=int, default=10000)
    p.add_argument('--steps', type=int, default=100_000)
    p.add_argument('--lesson-probability', type=float, default=None)
    p.add_argument('--size', type=int, default=None)
    p.add_argument('--algorithm', choices=['dfs', 'prim'], default='dfs')
    p.add_argument('--max-turns', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--model-path', type=Path, help='Also copy the saved model here (replaces an existing file)')
    p.add_argument('--resume', type=Path, help='Previous run directory with model.zip and config.json')
    args = p.parse_args()
    if args.save_every < 1:
        p.error('save-every must be positive')
    if args.auto_resume and (not args.cloud or args.resume):
        p.error('auto-resume requires cloud and cannot be combined with resume')
    if args.cloud:
        check_access()
        sync_pending()
        if args.auto_resume:
            args.resume = download_latest(prefix="ppo-v3-")
    if args.steps < 1 or args.threads < 1:
        p.error('steps and threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; use --device cpu or install CUDA-enabled PyTorch')
    torch.set_num_threads(args.threads)
    saved = json.loads((args.resume / 'config.json').read_text()) if args.resume else None
    if saved and saved.get('profile') != PROFILE:
        p.error('Earlier checkpoints are archived and cannot resume under v3 observations. Start a new model.')
    size = args.size or (saved['environment']['size'] if saved else 64)
    max_turns = args.max_turns or (saved['environment']['max_turns'] if saved and args.size is None else max(196, 4*size*size))
    config = (replace(Config(**saved['environment']), size=size, max_turns=max_turns, charge_turns=2) if saved else
              Config(size=size, algorithm=args.algorithm, max_turns=max_turns, charge_turns=2))
    output = args.output or Path('runs') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    lesson_probability = args.lesson_probability if args.lesson_probability is not None else (saved.get('lesson_probability', 0.3) if saved else 0.3)
    if not 0 <= lesson_probability <= 1:
        p.error('lesson-probability must be between 0 and 1')
    env = make_env(config, PROFILE, lesson_probability)
    check_env(env, warn=True)
    env = Monitor(env, str(output / 'monitor.csv'), info_keywords=('is_success', 'turns'))
    model = (PPO.load(args.resume / 'model.zip', env=env, device=args.device)
             if args.resume else PPO('MultiInputPolicy', env, n_steps=128, batch_size=64,
                                     n_epochs=4, learning_rate=3e-4, ent_coef=0.02,
                                     policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                                     seed=args.seed, device=args.device, verbose=1))
    cloud_tag = 'ppo-v3-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8] if args.cloud else None
    (output / 'config.json').write_text(json.dumps({'environment': asdict(config),
        'profile': PROFILE, 'lesson_probability': lesson_probability, 'cloud_tag': cloud_tag, 'seed': args.seed, 'requested_steps': args.steps, 'device': str(model.device),
        'resume': str(args.resume) if args.resume else None}, indent=2), encoding='utf-8')
    callback = Checkpoints(output, args.save_every, args.model_path, cloud_tag)
    start_steps = model.num_timesteps
    started = time.perf_counter()
    interrupted = False
    try:
        model.learn(total_timesteps=args.steps, reset_num_timesteps=not bool(args.resume), callback=callback)
    except KeyboardInterrupt:
        interrupted = True
        print('Interrupted: saving current model.')
    finally:
        callback.save()
        env.close()
    elapsed = time.perf_counter() - started
    summary = {'steps_this_run': model.num_timesteps - start_steps,
               'total_steps': model.num_timesteps, 'seconds': elapsed, 'interrupted': interrupted}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    print('Saved:', output.resolve())
    if args.cloud and callback.last_uploaded != model.num_timesteps:
        raise SystemExit('Training saved locally, but GitHub upload failed. Run sync_ppo.bat to retry.')


if __name__ == '__main__':
    main()
