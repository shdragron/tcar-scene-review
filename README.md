# tcar-scene-review

nuScenes 형식으로 변환한 tcar 주행 데이터를 펼쳐 보고, 규칙이 제안한 삭제 scene을 사람이 눈으로 확인해 결정하는 도구입니다.
규칙은 제안만 하고, 최종 결정은 사람이 합니다.

## 흐름

1. **규칙 제안** `python tools/curate.py`
   같은 장소에 멈춘 scene들 중 LiDAR diff(0.5초 간격 프레임 사이 바뀐 칸 수)가 가장 큰 것만 남기고 나머지를 삭제 제안합니다.
   결과: `selection/result.json`
2. **검토** `python tools/sample_viewer.py` → http://localhost:8765
   - 카메라 6대, LiDAR 3D(마우스 회전), 경로 지도, 실시간 재생
   - 검토 화면: 남긴 scene과 삭제 제안 scene을 나란히 재생(카메라 / LiDAR), scene마다 남기기·버리기
   - 결정: `selection/human_decisions.json`, 기록: `selection/decisions_log.txt`
   - 최종 목록(규칙 제안 + 사람 결정): `selection/final_selection.json`
3. **삭제 확정** 화면의 "삭제 확정" 버튼 또는 `python tools/apply_selection.py --apply`
   - 삭제 scene의 파일·표 기록을 `_removed/<시각>/`로 옮기고(지우지 않음), 남은 scene 이름을 빈 번호 없이 당깁니다
   - 되돌리기: `python tools/apply_selection.py --undo _removed/<시각>`

## 데이터 위치

도구는 기본으로 `tools/`의 상위 폴더를 데이터 루트로 씁니다(`--dataroot`로 변경).

```
<dataroot>/
  v1.0-trainval/   samples/   sweeps/   can_bus/   maps/   *.import.json
  selection/       # 규칙 결과, 사람 결정, 캐시 (git 제외)
  tools/           # 이 저장소
```

## 파일

| 파일 | 역할 |
|---|---|
| `tools/sample_viewer.py` / `.html` | 검토 뷰어 서버와 화면 (`/full`: 이전의 자세한 뷰어) |
| `tools/curate.py` | 삭제 제안 규칙 (`--mode stops` 기본, `--mode quota` 이전 2단계 방식) |
| `tools/apply_selection.py` | 최종 목록 적용 / 되돌리기 |
| `tools/scene_select.py` | 장소·자차 상태 특징, LiDAR diff |
| `tools/scene_risk.py` | 위험 이벤트 추정 (실제 주행 경로 위 TTC 등) |
| `tools/scene_objects.py`, `stop_events.py`, `scene_embed.py` | 객체 수, 정차 중 사건, CLIP 임베딩 (quota 모드용) |

필요 패키지: Python 3, numpy, opencv-python (LiDAR diff), torch · torchvision · open_clip (quota 모드 분석용).
