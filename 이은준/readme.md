# 이은준의 폴더

비기거나, 지는 경우 평균스텝에서 카운트가 안되는 것을 발견
- 어떻게 하면 평균스텝을 적게할 수 있을까?
  - 이기는 맵을 확실히 이기자!
    - 맵의 공통적인 부분을 찾자!
      - 단순한 맵이 이기기가 쉽다!


---

액션 맵 수정 후

---

olympics/object.py 수정(벽 부딛치면 각도 변환) - 벽 뚫음
  - 현재 상태에서 가장 최적의 행동을 계산하는 함수 사용
---

olympics/object.py -2 수정 - 오히려 줄음
 - 벽 뚫는 메소드 변경
---

agents/rl/submission 수정( Dana-Q 탐색 제어를 넣기 )
- DynaQAgent를 import
- Dyna-Q 에이전트가 학습할 환경을 정의
- submission.py에서 Dyna-Q 에이전트를 초기화
- Dyna-Q 에이전트를 학습시키는 루프를 작성
- 에이전트 성능을 평가

> 결과값이 같게 나옴
---
agents/rl/submission 수정( 탐험적 선택 활용 )
- 탐험 확률 초기화 (epsilon=0.1)
- choose_action 함수로 epsilon값에 따른 랜덤or최적을 선택

> 오히려 최적이 아닌 경로를 가 삐꾸가 많이 남

---

olympics/core.py 수정( 벽에 3번 부딛치면 180도 회전 )
- get_join_actions 메서드:
  - 상태에 따라 각 에이전트의 행동을 생성.
  - rl_agent를 통해 강화학습 에이전트의 행동을 선택.
- run_game 함수:
  - 게임 환경을 실행하고 에피소드 보상, 승리 수, 스텝 수를 기록.
  - 환경과 에이전트 목록을 입력받아 테스트를 진행.

> 알고리즘 확인밖에 안됨

----

rl_trainer/algo/ppo.py 수정( 하이퍼 파라미터값 조정  ) - 같음

---

submission의 벽 3초동안 부딛칠 때 방향 180도 변경
- collision_time_tracker 변수를 추가하여 벽과 충돌한 시간을 기록.
- 2초 후 속도 반전: 충돌이 2초 이상 유지되면, 속도를 반전
- collision_time_tracker를 초기화하는 메서드를 추가하여 필요시 타이머를 리셋할 수 있도록 함
- collision_response 메서드 내에서 벽 충돌 후 2초가 경과했을 때 속도를 반전시키는 로직을 추가

> 오류 떠서 포기

---
