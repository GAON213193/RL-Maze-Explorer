# 기반 구현 실행 기록 (2026-10-24)

실제 실행:
- `python -m unittest discover -s tests -v`: 19개 테스트 통과.
- `python -m environment.demo --headless`: 64×64 DFS, seed 42. 11턴 후 충전 1회,
  에너지 525/525, 충돌 0. 기준 최소 턴 477, 기준 에너지 420.
- `python -m environment.demo --headless --algorithm prim`: 동일 조건 Prim. 11턴 후 충전 1회,
  에너지 166.5/166.5, 충돌 0. 기준 최소 턴 149, 기준 에너지 133.
- `python -m environment.maze_generator --size 64 --algorithm dfs --seed 42 --quiet`:
  시작점 (63,23), 출구 (0,23), 최소 이동 칸 수 363.

테스트 범위:
외곽 배치·비모서리·서로 다른 변·연결성과 최소 길이·시드 재현성,
벽 편집 거부, 에너지/회전/충돌, 충전과 기억 보존, 충전 중 시간 제한,
숨겨진 출구 미노출, 관측 복사본, 벽/모서리 차폐, 시야 회전,
에너지 부족·정확한 0 에너지 도착·리셋, 방향을 포함한 oracle 경로 실행.

위 결과는 고정 행동을 실행한 환경 검사다. 강화학습 성능이나 탈출 성공률이 아니다.
데스크톱 GUI는 Tkinter import까지 확인했으나 이 환경에는 화면 서버가 없어
실제 창 표시·키보드 조작은 검증하지 않았다. Windows BAT 역시 Windows에서 실행하지 않았다.
PPO 학습, 사용자 GPU 실행, 웹 추론은 아직 수행하지 않았다.
