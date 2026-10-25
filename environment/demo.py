"""Manual desktop viewer, not a trained AI. No external Python packages needed."""
import argparse
import json
from .robot import Config,FORWARD,LEFT,RIGHT,CHARGE,UNKNOWN
from .robot_v3 import RangeRobotEnv as RobotEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--algorithm',choices=['dfs','prim'],default='dfs')
    parser.add_argument('--battery-margin',type=float,default=1.25)
    parser.add_argument('--charge-turns',type=int,default=2)
    parser.add_argument('--headless',action='store_true',help='Run deterministic smoke check, not learning')
    args=parser.parse_args()
    env=RobotEnv(Config(size=args.size,algorithm=args.algorithm,view_distance=3,
                        battery_margin=args.battery_margin,charge_turns=args.charge_turns))
    env.reset(args.seed)
    if args.headless:
        for action in (RIGHT,LEFT,FORWARD,RIGHT,RIGHT,FORWARD,CHARGE):
            if env.terminated or env.truncated:break
            env.step(action)
        print(json.dumps(env.debug_stats(),indent=2));return
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        parser.exit(1,'Tkinter is required for the viewer. Install Python with Tcl/Tk, or use --headless.\n')
    root=tk.Tk();root.title('RL Maze Explorer - manual environment test (not trained AI)')
    cell=max(4,min(24,512//args.size));extent=cell*args.size
    top=tk.Frame(root);top.pack()
    canvases=[]
    for title in ['전체 미로 (평가자 전용)','로봇 관측·기억']:
        frame=tk.Frame(top);frame.pack(side='left',padx=8,pady=8)
        tk.Label(frame,text=title).pack()
        canvas=tk.Canvas(frame,width=extent,height=extent,bg='white');canvas.pack();canvases.append(canvas)
    status=tk.StringVar();tk.Label(root,textvariable=status).pack()
    tk.Label(root,text='수동 조작 테스트입니다. ↑ 전진 / ← 좌회전 / → 우회전 / C 충전 / R 초기화').pack()
    def draw():
        obs=env.observation()
        for index,c in enumerate(canvases):
            c.delete('all')
            for y in range(env.maze.size):
                for x in range(env.maze.size):
                    v=env.maze.grid[y][x] if index==0 else obs['memory'][y][x]
                    color={-1:'#555b66',0:'#f5f5f5',1:'#20252b',2:'#67cf72',3:'#f5c84c'}[v]
                    if index==0:
                        if (x,y)==env.maze.goal:color='#67cf72'
                        if (x,y)==env.maze.start:color='#f5c84c'
                    if index==1 and (x,y) in env.visible and v==0:color='#c9e7ff'
                    c.create_rectangle(x*cell,y*cell,(x+1)*cell,(y+1)*cell,fill=color,outline='')
            trail=[((x+.5)*cell,(y+.5)*cell) for x,y in env.trajectory]
            if len(trail)>1:c.create_line(*[v for pt in trail for v in pt],fill='#e89645',width=1)
            x,y=env.position;cx,cy=(x+.5)*cell,(y+.5)*cell
            dx,dy=((0,-1),(1,0),(0,1),(-1,0))[env.direction]
            c.create_oval(cx-cell*.4,cy-cell*.4,cx+cell*.4,cy+cell*.4,fill='#e74646',outline='')
            c.create_line(cx,cy,cx+dx*cell*.6,cy+dy*cell*.6,fill='#275ddb',width=2)
        status.set(f'턴 {env.turns}  |  배터리 {env.battery:g}/{env.capacity:g}  |  충전 {env.charges}  |  충돌 {env.collisions}  |  {env.reason or "진행 중"}')
    def act(action):
        if not env.terminated and not env.truncated:env.step(action);draw()
    def reset():env.reset(args.seed);draw()
    bar=tk.Frame(root);bar.pack(pady=6)
    for label,a in [('좌회전',LEFT),('전진',FORWARD),('우회전',RIGHT),('충전',CHARGE)]:
        tk.Button(bar,text=label,command=lambda a=a:act(a)).pack(side='left',padx=3)
    tk.Button(bar,text='같은 미로 초기화',command=reset).pack(side='left')
    for key,a in [('Up',FORWARD),('Left',LEFT),('Right',RIGHT),('c',CHARGE)]:root.bind('<'+key+'>',lambda e,a=a:act(a))
    root.bind('<r>',lambda e:reset())
    draw();root.mainloop()

if __name__=='__main__':main()
