"""Replay a saved PPO policy; never train or mutate its weights."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from environment.gym_env import MazeGymEnv
from environment.robot import Config, DIRECTIONS
from .cloud import download_latest


def load_session(model_path, config_path, seed):
    config = Config(**json.loads(config_path.read_text(encoding='utf-8'))['environment'])
    env = MazeGymEnv(config)
    model = PPO.load(model_path, env=env, device='cpu')
    obs, _ = env.reset(seed=seed)
    return model, env, obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--local', action='store_true', help='Use root local backup instead of GitHub')
    p.add_argument('--model', type=Path, default=Path('ppo_model.zip'))
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--headless', type=int, default=0, help='Run this many actions without a window')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.local:
        model_path, config_path = args.model, args.model.with_suffix('.json')
    else:
        folder = download_latest()
        if folder is None:
            raise SystemExit('No GitHub checkpoint yet. Train first, or use --local.')
        model_path, config_path = folder/'model.zip', folder/'config.json'
    model, env, obs = load_session(model_path, config_path, args.seed)
    if args.headless:
        for _ in range(args.headless):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated: break
        print(json.dumps(env.robot.debug_stats())); env.close(); return
    import tkinter as tk
    root = tk.Tk(); root.title('PPO model replay - ' + str(model_path))
    size = env.config.size; cell = max(4, min(24, 512//size)); extent = cell*size
    row = tk.Frame(root); row.pack()
    canvases = []
    for title in ['전체 미로 · 평가자 화면', '로봇의 관측과 기억']:
        frame = tk.Frame(row); frame.pack(side='left', padx=6)
        tk.Label(frame, text=title).pack()
        canvas = tk.Canvas(frame, width=extent, height=extent); canvas.pack(); canvases.append(canvas)
    status = tk.StringVar(); tk.Label(root, textvariable=status).pack()
    running = False; finished = False; seed = args.seed
    delay = tk.IntVar(value=100)
    stochastic = tk.BooleanVar(value=False)
    def draw():
        robot = env.robot
        for index, canvas in enumerate(canvases):
            canvas.delete('all')
            for y in range(size):
                for x in range(size):
                    value = robot.maze.grid[y][x] if index == 0 else robot.memory[y][x]
                    color = {-1:'#555b66',0:'#f5f5f5',1:'#20252b',2:'#67cf72',3:'#f5c84c'}[value]
                    if index == 0:
                        if (x,y) == robot.maze.start: color = '#f5c84c'
                        if (x,y) == robot.maze.goal: color = '#67cf72'
                    elif (x,y) in robot.visible and value == 0: color = '#c9e7ff'
                    canvas.create_rectangle(x*cell,y*cell,(x+1)*cell,(y+1)*cell,fill=color,outline='')
            points = [v for x,y in robot.trajectory for v in ((x+.5)*cell,(y+.5)*cell)]
            if len(points)>2: canvas.create_line(*points, fill='#e89645')
            x,y = robot.position; cx,cy = (x+.5)*cell,(y+.5)*cell
            dx,dy = DIRECTIONS[robot.direction]
            canvas.create_oval(cx-cell*.4,cy-cell*.4,cx+cell*.4,cy+cell*.4,fill='#e74646')
            canvas.create_line(cx,cy,cx+dx*cell*.6,cy+dy*cell*.6,fill='#275ddb',width=2)
        status.set(f'시드 {seed} | 턴 {robot.turns} | 배터리 {robot.battery:g}/{robot.capacity:g} | 충전 {robot.charges} | {robot.reason or "진행 중"}')
    def step():
        nonlocal obs, finished, running
        if finished: return
        action, _ = model.predict(obs, deterministic=not stochastic.get())
        obs, _, terminated, truncated, _ = env.step(int(action))
        finished = terminated or truncated
        if finished: running = False
        draw()
    def toggle():
        nonlocal running
        running = not running and not finished
    def reset(new=False):
        nonlocal obs, finished, running, seed
        running = False; finished = False
        if new: seed += 1
        obs, _ = env.reset(seed=seed); draw()
    bar = tk.Frame(root); bar.pack()
    for label, command in [('재생 / 일시정지',toggle),('한 행동',step),('같은 미로',reset),('다음 미로',lambda:reset(True))]:
        tk.Button(bar,text=label,command=command).pack(side='left')
    tk.Checkbutton(root,text='확률적으로 행동 선택 (끄면 가장 높은 확률의 행동)',variable=stochastic).pack()
    tk.Scale(root,from_=10,to=1000,resolution=10,orient='horizontal',label='행동 간격(ms)',variable=delay).pack(fill='x')
    def tick():
        if running: step()
        root.after(delay.get(),tick)
    draw(); tick(); root.mainloop(); env.close()

if __name__ == '__main__':
    main()
