"""Human-readable summaries with explicit cohort sizes and denominators."""
import csv
import json
from pathlib import Path


def rate(count, total):
    return f'{100*count/total:.1f}% ({count}/{total})' if total else '집계된 판 없음'


def training_report(output, metrics):
    config = json.loads((output/'config.json').read_text(encoding='utf-8'))
    env = config['environment']
    lines = ['학습 중 결과 요약 (별도 평가 결과가 아닙니다)',
             f"설정: {config.get('profile', 'legacy-v1')} / {env['size']}×{env['size']} / 충전 {env['charge_turns']}턴",
             f"누적 학습 스텝: {metrics['total_steps']:,}",
             '아래 통계는 이번 실행에서 완료된 판만 집계합니다. 이어 학습 이전 판은 포함하지 않습니다.',
             '전체 성공률(일반/연습 혼합): '+rate(metrics['successes'],metrics['episodes']),
             f"배터리 부족 종료: {metrics['battery_depletions']}판",
             f"제자리 중단: {metrics['stalls']}판"]
    recent = metrics.get('recent_100_success_rate')
    lines.append('최근 최대 100판 성공률: '+(f'{recent*100:.1f}%' if recent is not None else '집계된 판 없음'))
    row = {'total_steps':metrics['total_steps'], 'episodes':metrics['episodes'],
           'successes':metrics['successes'], 'success_rate':metrics['success_rate'],
           'recent_100_success_rate':recent, 'battery_depletions':metrics['battery_depletions'],
           'stalls':metrics['stalls'], 'mean_turns':metrics['mean_turns']}
    for key,label in [('navigation','일반 탈출'),('return_practice','귀환·충전 연습')]:
        stats=metrics.get('scenarios',{}).get(key,{})
        n=stats.get('episodes',0);success=stats.get('successes',0);charged=stats.get('episodes_with_charge',0)
        lines.extend(['',label, '탈출 성공률: '+rate(success,n),
                      '시작점 귀환: '+rate(stats.get('returns',0),n),
                      f"실제 충전: {stats.get('charges',0)}회 / 충전한 판 {charged}판",
                      f"귀환 후 충전한 판: {stats.get('charged_after_return',0)}판",
                      '충전한 판 중 탈출 성공률: '+rate(stats.get('success_after_charge',0),charged)])
        row[key+'_episodes']=n
        row[key+'_success_rate']=success/n if n else None
        row[key+'_charged_episodes']=charged
        row[key+'_success_after_charge']=stats.get('success_after_charge',0)
    lines.extend(['','평균 소요 턴(성공·실패 포함): '+(f"{metrics['mean_turns']:.2f}" if metrics['mean_turns'] is not None else '집계된 판 없음'),
                  '상세 판별 기록: episodes.csv', '학습 진행 추이: training_history.csv'])
    (output/'training_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    history=output/'training_history.csv'
    rows=[]
    if history.exists():
        with history.open(encoding='utf-8-sig',newline='') as f:
            rows=[r for r in csv.DictReader(f) if int(r['total_steps']) != metrics['total_steps']]
    rows.append(row)
    with history.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)


def evaluation_report(path, results, metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    meta=path.with_suffix('.metadata.json')
    meta.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['별도 미로 평가 결과 (가중치 갱신 없음)',
           f"모델 누적 스텝: {metadata['total_steps']:,}",f"모델 SHA256: {metadata['model_sha256']}",
           f"설정: {metadata['profile']} / 요청 상황: {metadata['scenario']}",
           '각 행동 선택 방식은 동일한 평가 시드 집합을 사용합니다. 두 방식의 판수를 합산해 성공률을 계산하지 않습니다.']
    rows=[]
    for result in results:
        n=result['episodes'];outcomes=result['outcomes'];charging=result['charging']
        success=outcomes.get('success',0);charged=charging.get('episodes_with_charge',0)
        mode='최대 확률 행동' if result['deterministic'] else '확률적 행동'
        lines.extend(['',mode, f"미로: {result['size']}×{result['size']} / 시드: {result['seed']}부터 {n}개",
                      '탈출 성공률: '+rate(success,n),
                      f"평균 소요 턴(성공·실패 포함): {result['mean_turns']:.2f}",
                      f"배터리 부족: {outcomes.get('battery_depleted',0)+outcomes.get('insufficient_energy',0)}판",
                      f"제자리 중단: {outcomes.get('stalled',0)}판",
                      f"실제 충전: {charging.get('total_charges',0)}회 / 충전한 판 {charged}판",
                      '충전한 판 중 탈출 성공률: '+rate(charging.get('success_after_charge',0),charged),
                      '실제 상황 구성: '+json.dumps(result['scenarios'],ensure_ascii=False)])
        rows.append(dict(mode=mode,size=result['size'],seed=result['seed'],episodes=n,successes=success,
                         success_rate=success/n,mean_turns=result['mean_turns'],
                         battery_depletions=outcomes.get('battery_depleted',0)+outcomes.get('insufficient_energy',0),
                         charged_episodes=charged,success_after_charge=charging.get('success_after_charge',0)))
    path.with_suffix('.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    with path.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return [path,path.with_suffix('.txt'),path.with_suffix('.csv'),meta]
