# ML-Contest
# 기본 점수 : 138.685185....

+-----------+--------+-------------------+

|   Name    | random |        rl         |

+-----------+--------+-------------------+

|   score   |  6.0   |       54.0        |

|    win    |  6.0   |       54.0        |

| avg_steps |   -    | 138.6851851851852 |

+-----------+--------+-------------------+


# 주최쪽 URL
> https://www.pygame.org/wiki/Contribute

# 사용 방법
이은준의 경우
C드라이브 위의 Competition_Olympics-Running 폴더에서
```
conda activate Contest 
```
-> 이은준의 conda 가상환경 이름이 Contest임
```
pip install –r requirements.txt
```
> 다운로드 이미 했으면 안해도 됨
```
python run_log.py --my_ai "rl" --opponent "random"
```
> AssertionError: None 뜰 수 있는데 붉은 색이 아닌 하얀 색 AssertionError: None이면 상관없이 다음꺼 진행
```  
python evaluation_local.py --my_ai rl --opponent random --episode=100 --map=all
```
-----------
원본 깃허브 ![https://github.com/jidiai/Competition_Olympics-Integrated]

---
# 회의 내용
## 기계학습
1. 교수님이 말한 숫자 수정 대안
2. 전체적인 수정(코드)
