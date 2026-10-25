"""Fixed-seed evaluation, separate from training. No weight updates."""
import argparse
import json
from pathlib import Path
from collections import Counter
import torch
from stable_baselines3 import PPO
from .profiles import make_env
from environment.robot import Config


def evaluate(model, env, episodes=30, seed=10000, deterministic=True):
    outcomes=Counter(); actions=Counter(); cells=[]; turns=[]; moves=[]
    for i in range(episodes):
        torch.manual_seed(seed+i)
        obs,_=env.reset(seed=seed+i)
        for _ in range(env.config.max_turns):
            action,_=model.predict(obs,deterministic=deterministic)
            actions[int(action)]+=1
            obs,_,terminated,truncated,info=env.step(int(action))
            if terminated or truncated: break
        outcomes[info['reason'] or 'evaluation_limit']+=1
        turns.append(env.robot.turns)
        cells.append(len(set(env.robot.trajectory)))
        moves.append(sum(a!=b for a,b in zip(env.robot.trajectory,env.robot.trajectory[1:])))
    return dict(episodes=episodes,seed=seed,deterministic=deterministic,size=env.config.size,
                success_rate=outcomes['success']/episodes,outcomes=dict(outcomes),actions=dict(actions),
                mean_unique_cells=sum(cells)/episodes,mean_moves=sum(moves)/episodes,
                mean_turns=sum(turns)/episodes)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--output',type=Path)
    p.add_argument('--untrained',action='store_true')
    args=p.parse_args()
    if args.episodes < 1: p.error('episodes must be positive')
    torch.set_num_threads(1)
    config=json.loads((args.run/'config.json').read_text())
    env=make_env(Config(**config['environment']),config.get('profile','legacy-v1'))
    model=(PPO('MultiInputPolicy',env,seed=42,device='cpu',
               policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128]))) if args.untrained else
           PPO.load(args.run/'model.zip',env=env,device='cpu'))
    result=[evaluate(model,env,args.episodes,args.seed,d) for d in (True,False)]
    print(json.dumps(result,indent=2))
    if args.output: args.output.write_text(json.dumps(result,indent=2))
    env.close()

if __name__=='__main__':main()
