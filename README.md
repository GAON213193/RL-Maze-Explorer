# RL-Maze-Explorer

강화학습 기반 동적 환경 적응형 의사결정 모델 개발 (대학 인공지능응용 프로젝트)

## 목표
사전 학습된 에이전트가 **처음 보는 미로**에서, 전체 지도와 출구 위치를 전달받지 않은 채 제한된 시야와 탐색 기억을 활용하여 출구를 찾도록 한다.

- 기본 크기: 32×32, 확장 실험: 64×64
- 행동: 전진, 왼쪽 90도 회전, 오른쪽 90도 회전 (각 1턴)
- 관측: 지도 크기, 현재 위치·방향, 벽에 차단되는 제한된 시야
- 기억: 관찰한 벽·통로를 기록하되 미관측 지역은 알 수 없음
- 학습: 다양한 절차 생성 미로에서 오프라인으로 수행
- 평가: 학습에 사용하지 않은 미로에서 **모델 가중치를 변경하지 않고** 수행
- 웹: 추후 학습된 모델의 실행·관찰·미로 편집 전용 (학습 금지)

## 현재 구현 범위
초기 버전에는 **독립적인 미로 생성기, BFS 경로 검증, 텍스트 미로 표시, 자동 테스트**가 포함된다. 아직 강화학습 모델, 방향별 시야, 기억 지도, 웹 UI는 구현되지 않았다.

## 실행
Python 3.10+ 권장. 초기 환경 생성기는 외부 패키지가 필요 없다.

```bash
python -m environment.maze_generator --size 32 --algorithm dfs --seed 42
python -m environment.maze_generator --size 64 --algorithm prim --seed 7 --quiet
python -m unittest discover -s tests -v
```

## 개발 순서
1. 미로 생성 및 연결성 검사 **(초기 구현)**
2. 이동/회전과 가시선(LOS), 탐색 기억
3. Gymnasium 환경과 보상/에피소드 종료
4. PPO 초기 학습, 고정 모델 평가, 결과 CSV/그래프
5. 좌측 전체 미로/우측 에이전트 인식 지도 웹 테스트
6. 동적 벽 변경과 현재 위치→출구 연결성 검사

## 참고 연구·프로젝트
- Parisotto & Salakhutdinov, *Neural Map: Structured Memory for Deep Reinforcement Learning* (ICLR 2018)
- Hausknecht & Stone, *Deep Recurrent Q-Learning for Partially Observable MDPs* (2015)
- Burda et al., *Exploration by Random Network Distillation* (2018)
- Farama Foundation, *MiniGrid* (구현 방식 참고 후보)

기존 연구의 **아이디어와 적합한 구현 방식만 선별적으로 참고**한다. 외부 코드 재사용 시 해당 라이선스와 출처를 확인한다.
