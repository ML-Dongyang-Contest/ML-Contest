# 이은준의 폴더

비기거나, 지는 경우 평균스텝에서 카운트가 안되는 것을 발견
- 어떻게 하면 평균스텝을 적게할 수 있을까?
  - 이기는 맵을 확실히 이기자!
    - 맵의 공통적인 부분을 찾자!
      - 단순한 맵이 이기기가 쉽다!



---

`2024-11-27`
### 액션 맵 수정 후

---

`2024-12-27`
### olympics/object.py 수정(벽 부딛치면 각도 변환)
  - 현재 상태에서 가장 최적의 행동을 계산하는 함수 사용

> 벽 뚫음

---

`2024-11-27`
### olympics/object.py -2 수정
 - 벽 뚫는 메소드 변경

> 오히려 줄음

---

`2024-11-27`
### agents/rl/submission 수정( Dana-Q 탐색 제어를 넣기 )
- DynaQAgent를 import
- Dyna-Q 에이전트가 학습할 환경을 정의
- submission.py에서 Dyna-Q 에이전트를 초기화
- Dyna-Q 에이전트를 학습시키는 루프를 작성
- 에이전트 성능을 평가


---

`2024-12-4`
### agents/rl/submission 수정( 탐험적 선택 활용 )
- 탐험 확률 초기화 (epsilon=0.1)
- choose_action 함수로 epsilon값에 따른 랜덤or최적을 선택

> 오히려 최적이 아닌 경로를 가 삐꾸가 많이 남

---

`2024-11-29`
### olympics/core.py 수정( 벽에 3번 부딛치면 180도 회전 ), 벽 회피 기능
- get_join_actions 메서드:
  - 상태에 따라 각 에이전트의 행동을 생성.
  - rl_agent를 통해 강화학습 에이전트의 행동을 선택.
- run_game 함수:
  - 게임 환경을 실행하고 에피소드 보상, 승리 수, 스텝 수를 기록.
  - 환경과 에이전트 목록을 입력받아 테스트를 진행.



----

`2024-12-05`
### rl_trainer/algo/ppo.py 수정( 하이퍼 파라미터값 조정 )
- clip_param:
  - PPO 알고리즘에서 정책 업데이트를 제한하기 위해 사용되는 클리핑 파라미터.
  - 기존 정책과 새 정책의 차이를 일정 범위 내로 제한해 학습 안정성을 높이는 데 사용
- max_grad_norm:
  - 그래디언트 클리핑에 사용되는 최대 값 그래디언트 폭발 문제를 방지하기 위해 설정
- ppo_update_time:
  - PPO 알고리즘에서 정책을 몇 번 업데이트할지 설정하는 값
  -한 번의 학습 반복(iteration)에서 정책을 얼마나 자주 업데이트할지 결정
- buffer_capacity:
  - Replay Buffer의 최대 용량
  - 강화학습에서 수집된 경험 데이터를 저장하는 공간 크기를 의미
- batch_size:
  - 학습에 사용할 미니배치 크기
  - 네트워크 업데이트 시, 한 번에 처리할 데이터의 수를 의미
- gamma:
  - 할인율로, 미래 보상에 대한 현재 가치의 중요도를 조정함
  - 0.99라면 미래 보상을 많이 반영하는 설정임
- lr:
  - 학습률(Learning Rate)로, 네트워크의 가중치를 업데이트할 때 변화 크기를 결정
- action_space:
  - 에이전트가 선택할 수 있는 행동(action)의 수
  - 여기서는 36개의 행동 공간이 설정되어 있지만, 주석 처리된 action_space = 3은 특정 상황에서 행동 공간이 더 적을 수도 있음을 암시
- state_space:
  - 환경에서의 상태(state)를 표현하는 차원의 크기
  - 이 값이 625라면, 총 625개의 상태로 구성된 환경에서 학습한다는 뜻 

+ clip_param이 0.7, ppo_update_time가 30, batch_size가 16, lr가 0.001 일 경우 
+ clip_param이 0.1, ppo_update_time가 50, batch_size가 8, lr가 0.01일 경우 
+ clip_param이 0.3, ppo_update_time가 70, batch_size가 8, lr가 0.0001일 경우 

---

`2024-12-06`
### submission의 벽 3초동안 부딛칠 때 방향 180도 변경
- collision_time_tracker 변수를 추가하여 벽과 충돌한 시간을 기록.
- 2초 후 속도 반전: 충돌이 2초 이상 유지되면, 속도를 반전
- collision_time_tracker를 초기화하는 메서드를 추가하여 필요시 타이머를 리셋할 수 있도록 함
- collision_response 메서드 내에서 벽 충돌 후 2초가 경과했을 때 속도를 반전시키는 로직을 추가



---

`2024-12-07`
### rl_trainer/algo/ppo.py 수정( 손실 함수 재지정 )
- update 함수의 action_loss를 기존의 최소 클리핑 방식에서 KL-Divergence 를 추가
```
# 변경 전
ratio = (action_prob / old_action_log_prob[index])
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
```

```
# 변경 후
ratio = (action_prob / old_action_log_prob[index])
surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

kl_div = F.kl_div(action_prob.log(), old_action_log_prob, reduction='batchmean')
action_loss = -torch.min(surr1, surr2).mean() + 0.01 * kl_div                       # 손실함수 변경
```

----

`2024-12-07`
#  rl_trainer/algo/ppo.py의 def save() 함수가 호출이 안 되는 것을 발견

실행 중에 자동으로 save()함수가 호출되지 않은 것을 발견 → 로직을 고쳐도 기존의 학습한 내용가지고만 감

로직을 고칠 필요가 없음

---

`2024-12-07`
# 자동 저장 save() 호출
```
#### ppo 클래스 외부에 생성
def get_next_run_dir(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    existing_runs = [d for d in os.listdir(base_path) if d.startswith("run") and d[3:].isdigit()]
    if existing_runs:
        next_run_num = max([int(d[3:]) for d in existing_runs]) + 1
    else:
        next_run_num = 6
    return os.path.join(base_path, f"run{next_run_num}")
#####
```
```
def update(self, i_ep):
    self.clear_buffer()
    # 자동 저장 호출
    save_path = "models/olympics-running"
    self.save(save_path=save_path, episode=i_ep)

def save(self, save_path, episode):
    run_dir = get_next_run_dir(save_path)  # 저장 디렉토리 자동 생성
    os.makedirs(run_dir, exist_ok=True)   # 디렉토리 생성
    model_actor_path = os.path.join(run_dir, f"actor_{episode}.pth")
    model_critic_path = os.path.join(run_dir, f"critic_{episode}.pth")
    torch.save(self.actor_net.state_dict(), model_actor_path)
    torch.save(self.critic_net.state_dict(), model_critic_path)
    print(f"Models saved in {run_dir}")
```


---

`2024-12-08`
# rl_trainner/main.py의 load_model 을 True로 해 이미 학습한 파일들 불러오기

```
if __name__ == '__main__':
    args = parser.parse_args()
    args.load_model = True    # 이부분이 원래 주석처리되있었음
    #args.load_run = 3
    #args.map = 3
    #args.load_episode= 900
    main(args)
```
같게 나와서 위 모두 주석을 없애봄

---

`2024-12-09`
# 애초에 오류가 있었음
```
python run_log.py --my_ai "rl" --opponent "random"
```
를 사용하면

![image](https://github.com/user-attachments/assets/9ca7543d-95f9-4204-adb6-50395c2dd17d)

오류가 떠서 

범위 벗어난 값 처리를 하기 위해 `action[1]`을 클리핑하는 방식을 사용
```
action[1] = np.clip(action[1], self.action_theta[0], self.action_theta[1])
```

오류는 안뜨지만 처음보는 광경을 봄

![image](https://github.com/user-attachments/assets/b607de19-459b-4a89-910d-4a8351877b83)

step를 한번 움직일 때 마다 저렇게 print됨

- 함수 호출을 따라가봤는데 무슨 소리인지 모르겠음
