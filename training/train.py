"""Run a bounded PPO job; save model and configuration for resuming."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from environment.gym_env import MazeGymEnv
from environment.robot import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100_000)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--algorithm', choices=['dfs', 'prim'], default='dfs')
    p.add_argument('--max-turns', type=int, default=20_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--resume', type=Path, help='Previous run directory with model.zip and config.json')
    args = p.parse_args()
    if args.steps < 1 or args.threads < 1:
        p.error('steps and threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; use --device cpu or install CUDA-enabled PyTorch')
    torch.set_num_threads(args.threads)
    config = (Config(**json.loads((args.resume / 'config.json').read_text())['environment'])
              if args.resume else Config(size=args.size, algorithm=args.algorithm, max_turns=args.max_turns))
    output = args.output or Path('runs') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    env = MazeGymEnv(config)
    check_env(env, warn=True)
    env = Monitor(env, str(output / 'monitor.csv'), info_keywords=('is_success', 'turns'))
    model = (PPO.load(args.resume / 'model.zip', env=env, device=args.device)
             if args.resume else PPO('MultiInputPolicy', env, n_steps=128, batch_size=64,
                                     n_epochs=4, learning_rate=3e-4, ent_coef=0.01,
                                     policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                                     seed=args.seed, device=args.device, verbose=1))
    (output / 'config.json').write_text(json.dumps({'environment': asdict(config),
        'seed': args.seed, 'requested_steps': args.steps, 'device': str(model.device),
        'resume': str(args.resume) if args.resume else None}, indent=2), encoding='utf-8')
    start_steps = model.num_timesteps
    started = time.perf_counter()
    interrupted = False
    try:
        model.learn(total_timesteps=args.steps, reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        interrupted = True
        print('Interrupted: saving current model.')
    finally:
        model.save(output / 'model.zip')
        env.close()
    elapsed = time.perf_counter() - started
    summary = {'steps_this_run': model.num_timesteps - start_steps,
               'total_steps': model.num_timesteps, 'seconds': elapsed, 'interrupted': interrupted}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    print('Saved:', output.resolve())


if __name__ == '__main__':
    main()
