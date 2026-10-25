# PPO 연결 및 로컬 학습

## Windows 더블클릭 테스트

전체 프로젝트를 압축 해제한 후 루트의 `test_ppo.bat`를 더블클릭합니다.
Python 3.12가 설치되어 있으면 가상환경 생성, 필요한 패키지 설치, 자동 검사, 짧은 학습을 순서대로 실행합니다.
첫 실행에는 인터넷 연결과 패키지 다운로드 시간이 필요합니다.
테스트 모델은 실행파일과 같은 루트의 `ppo_test_model.zip`에 저장하며 다음 테스트 때 덮어씁니다.
각 실행의 원본 모델·로그·이어 학습 설정은 별도 `runs` 폴더에 남습니다.
오류 또는 완료 후 창은 키를 누를 때까지 유지됩니다. Windows BAT 자체는 이 Linux 환경에서 실행 검증하지 못했습니다.

Python 3.12 가상환경을 권장합니다. 수동 데모는 기존처럼 외부 패키지 없이 실행할 수 있습니다.
PPO는 공식 stable_baselines3 패키지에서 가져옵니다. references의 원본 사본을 직접 실행하지 않습니다.

## Windows PowerShell: 프로젝트 루트에서 설치

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
```

Python 3.12가 없다면 먼저 설치해야 합니다. 이 명령은 CPU 버전이며 GPU 없이 학습할 수 있습니다.
이번 검증은 Linux CPU에서 수행했습니다. Windows 및 사용자 RTX 4080 SUPER에서의 실행은 아직 검증하지 않았습니다.
CUDA 사용은 호환 PyTorch 설치가 추가로 필요합니다. 기본 CPU 실행부터 확인하십시오.

## 연결 확인 (짧은 실행)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m training.train --size 64 --steps 256 --max-turns 128
```

짧은 테스트의 최대 턴 128은 기능 검증용이며 64×64 미로 탈출 성능을 평가할 설정이 아닙니다.
실행이 끝나면 출력된 runs 폴더에 model.zip, config.json, summary.json, monitor.csv가 저장됩니다.

## 본 학습 (사용자가 실행)

```powershell
.\.venv\Scripts\python.exe -m training.train --size 64 --steps 100000 --algorithm dfs
```

기본 최대 턴은 20,000입니다. 10만 스텝은 시작 예시이며 성공에 충분한 학습량을 의미하지 않습니다.
Ctrl+C로 중단하면 현재 모델을 저장합니다. 프로세스 강제 종료/전원 종료에는 저장을 보장하지 않습니다.
128 스텝 단위로 수집하므로 요청 스텝보다 최대 127 스텝 더 실행할 수 있습니다.

이어 학습하려면 실제 저장 폴더 경로를 지정합니다.

```powershell
.\.venv\Scripts\python.exe -m training.train --resume runs/실제저장폴더 --steps 100000
```

이어 학습은 기존 모델/옵티마이저와 환경 설정을 불러오되, 미로 에피소드는 새로 시작합니다.
--resume 사용 시 --size, --algorithm, --max-turns 대신 저장된 config.json 설정을 사용합니다.
저장된 모델은 동일한 미로 크기의 입력 구조를 요구합니다. 크기를 바꾸어 이어 학습하는 기능은 없습니다.
기존 결과 덮어쓰기를 막기 위해 --output에는 존재하지 않는 새 폴더를 지정해야 합니다.

## 정책 입력 및 학습 의미

- 기억 지도: 미관측/통로/벽/관측된 출구/충전소를 one-hot으로 표현
- 현재 시야 표시, 현재 좌표·시작 좌표·방향·배터리 비율·용량·직전 충돌
- 행동: 전진, 좌회전, 우회전, 충전
- 전체 미로와 숨겨진 출구 좌표 및 최단경로는 입력에서 제외
- 매 reset마다 새 시드로 미로 생성, seed를 명시하면 재현 가능

기존 배터리 산정 방식의 경로 비용 간접 정보는 그대로 존재합니다.
PPO의 gamma는 의사결정 1회 기준입니다. 충전은 여러 턴을 소모하며 보상에는 그 시간이 반영되지만,
시간별 할인율을 별도로 적용하는 semi-Markov 학습은 아닙니다.
관측 기억은 환경이 명시적으로 유지하며 모델 자체는 재귀 신경망이 아닌 MLP입니다.

## 검증 결과 (2026-10-25)

- Python 3.12.14, torch 2.14.1+cpu, SB3 2.9.0, Gymnasium 1.4.0, NumPy 2.5.4
- 기존 19개 + 학습 연결 2개 = 21개 unittest 통과
- DFS/Prim 각각 실제 64×64에서 Gymnasium/SB3 환경 검사 통과
- 각각 PPO 128 스텝 학습: 파라미터 변경 및 모든 파라미터 유한값 확인
- 모델 저장/로드 후 파라미터와 결정적 행동 일치, 추가 64 스텝 학습 통과
- 기본 학습 스크립트: 64×64 DFS, 256 스텝, 최대 128턴에서 약 1.94초 (모델 초기화 제외)
- 해당 짧은 실행 성공률 0: 연결 테스트이며 학습 성능을 입증하는 결과가 아닙니다.

학습 로그의 ep_len_mean은 행동 호출 횟수입니다. 충전은 여러 턴이므로 실제 시간은 monitor.csv의 turns를 사용합니다.
웹 추론 및 학습된 모델을 수동 데모에서 재생하는 기능은 아직 구현하지 않았습니다.
