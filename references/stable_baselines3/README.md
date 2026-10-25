# 공개 PPO 구현 참고 코드

Stable-Baselines3의 공식 PPO 구현을 수정 없이 가져왔습니다. 사전학습 가중치는 포함하지 않습니다.

- 원본 저장소: https://github.com/DLR-RM/stable-baselines3
- 원본 파일: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
- 수집일: 2026-10-25
- 원본 ppo.py Git blob SHA: 51863ffc1ee4c0d5d300a408f15f81158e347dfa
- 라이선스: MIT (동봉 LICENSE의 저작권 및 허가 고지를 보존)
- PPO 논문: https://arxiv.org/abs/1707.06347

## 용도

ppo.py는 알고리즘을 읽고 참고하기 위한 원본 사본입니다. 버퍼, 정책 신경망, 학습 루프 등의 구현을 stable_baselines3.common에서 가져오므로 이 파일 하나만으로 독립 실행할 수 없습니다.

실제 프로젝트 학습에는 패키지를 설치하고 공식 클래스를 가져와 사용하는 방식을 권장합니다.

```bash
python -m pip install stable-baselines3
```

```python
from stable_baselines3 import PPO
```

위 import는 패키지 설치 후의 사용 방법이며 미로 학습을 실행하지 않습니다. 프로젝트의 RobotEnv를 Gymnasium 인터페이스로 연결하고 관측·행동 공간을 정의하는 작업은 아직 남아 있습니다. 실제 연결 시 사용할 패키지 버전을 고정하고 동작을 검증해야 합니다. 여기 보관한 master 원본과 설치되는 릴리스 버전은 다를 수 있습니다.
