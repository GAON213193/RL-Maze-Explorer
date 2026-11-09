# 인공지능 모델 및 학습 기록

이 `models` 브랜치는 학습된 모델과 실험 기록을 확인하는 용도입니다. 실행 코드·EXE·Python 환경은 포함하지 않습니다.

코드와 실행 프로그램: [codex/browser-training-launcher](https://github.com/GAON213193/RL-Maze-Explorer/tree/codex/browser-training-launcher)

## 확인한 모델

| 실행 | 누적 스텝 | 모델 및 기록 |
|---|---:|---|
| 2026-11-09 1회차 | 100,096 | [폴더](checkpoints/ppo-v3-20261109-144234-dffa553a) |
| 2026-11-09 2회차(이어 학습) | 200,192 | [폴더](checkpoints/ppo-v3-20261109-150405-9f7b2042) |

[학습 분석](ANALYSIS.md) · [전체 보관 목록과 해시](manifest.json) · [최근 업로드 모델](latest.json)

각 폴더의 model.zip은 학습 가중치이고 config.json, metrics.json, CSV와 TXT는 설정·학습 기록입니다. 최신 코드는 앞으로 각 실행의 최근 체크포인트를 해당 폴더에 자동 반영합니다. 원본 체크포인트 이력은 [GitHub Releases](https://github.com/GAON213193/RL-Maze-Explorer/releases)에 보존합니다.

로컬 이어 학습에 사용하려면 원하는 실행 폴더를 코드 폴더의 runs 아래에 복사한 뒤 웹에서 해당 모델을 선택하세요. 코드 전체를 models 브랜치에서 받을 필요는 없습니다.
