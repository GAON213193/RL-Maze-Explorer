"""Fixed-seed evaluation, separate from training. No weight updates."""
import argparse
import json
import hashlib
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED
from .reports import evaluation_report
from .cloud import check_access, download_latest, publish
from pathlib import Path
from collections import Counter
import torch
from stable_baselines3 import PPO
from .profiles import make_env
from environment.robot import Config


def evaluate(model, env, episodes=30, seed=10000, deterministic=True):
    outcomes=Counter(); actions=Counter(); cells=[]; turns=[]; moves=[]; charging=Counter(); scenarios=Counter()
    for i in range(episodes):
        torch.manual_seed(seed+i)
        obs,_=env.reset(seed=seed+i)
        for _ in range(env.config.max_turns):
            action,_=model.predict(obs,deterministic=deterministic)
            actions[int(action)]+=1
            obs,_,terminated,truncated,info=env.step(int(action))
            if terminated or truncated: break
        scenarios[info.get('scenario', 'navigation')] += 1
        charging['episodes_with_charge'] += env.robot.charges > 0
        charging['total_charges'] += env.robot.charges
        charging['returns'] += bool(info.get('returned_to_station'))
        charging['charged_after_return'] += bool(info.get('charged_after_return'))
        charging['success_after_charge'] += info['is_success'] and env.robot.charges > 0
        outcomes[info['reason'] or 'evaluation_limit']+=1
        turns.append(env.robot.turns)
        cells.append(len(set(env.robot.trajectory)))
        moves.append(info.get('forward_moves', sum(a!=b for a,b in zip(env.robot.trajectory,env.robot.trajectory[1:]))))
    return dict(episodes=episodes,seed=seed,deterministic=deterministic,size=env.config.size,
                charging=dict(charging), scenarios=dict(scenarios), success_rate=outcomes['success']/episodes,outcomes=dict(outcomes),actions=dict(actions),
                mean_unique_cells=sum(cells)/episodes,mean_moves=sum(moves)/episodes,
                mean_turns=sum(turns)/episodes)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path)
    p.add_argument('--cloud',action='store_true',help='Download latest v3 if run is omitted and upload evaluation reports')
    p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--output',type=Path)
    p.add_argument('--practice', action='store_true', help='Evaluate return-practice starts separately')
    p.add_argument('--untrained',action='store_true')
    args=p.parse_args()
    if args.cloud:
        check_access()
        if args.run is None: args.run=download_latest(prefix='ppo-v3-')
    if args.run is None: p.error('Specify --run or provide an available GitHub v3 checkpoint with --cloud')
    if args.episodes < 1: p.error('episodes must be positive')
    torch.set_num_threads(1)
    config=json.loads((args.run/'config.json').read_text())
    env=make_env(Config(**config['environment']),config.get('profile','legacy-v1'), 1.0 if args.practice else 0.0)
    model=(PPO('MultiInputPolicy',env,seed=42,device='cpu',
               policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128]))) if args.untrained else
           PPO.load(args.run/'model.zip',env=env,device='cpu'))
    result=[evaluate(model,env,args.episodes,args.seed,d) for d in (True,False)]
    print(json.dumps(result,indent=2))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.output or args.run/'evaluations'/('evaluation-'+stamp+'.json')
    metadata=dict(total_steps=model.num_timesteps,profile=config.get('profile','legacy-v1'),
                  scenario='return_practice' if args.practice else 'navigation',
                  model_sha256=hashlib.sha256((args.run/'model.zip').read_bytes()).hexdigest() if not args.untrained else 'untrained-baseline',
                  source_run=str(args.run),source_release=config.get('cloud_tag'),created_at_utc=stamp,
                  environment=config['environment'],untrained=args.untrained)
    files=evaluation_report(output,result,metadata)
    print('Evaluation saved:',output.with_suffix('.txt').resolve())
    if args.cloud:
        tag=config.get('cloud_tag')
        if not tag: raise SystemExit('Evaluation saved locally; this run has no GitHub release tag.')
        bundle=output.parent/('evaluation-'+stamp+'.zip')
        with ZipFile(bundle,'w',compression=ZIP_DEFLATED) as z:
            for file in files:z.write(file,file.name)
        pending=bundle.with_suffix('.evaluation-pending.json')
        pending.write_text(json.dumps(dict(tag=tag,bundle=bundle.name)),encoding='utf-8')
        try:
            publish(bundle,tag)
            print('GitHub evaluation saved:',tag,bundle.name)
        except RuntimeError as exc:
            raise SystemExit('Evaluation saved locally; upload failed. Run sync_ppo.bat to retry. '+str(exc))
    env.close()

if __name__=='__main__':main()
