# ML-Contest

# 사용 방법
이은준의 경우
C드라이브 위의 Competition_Olympics-Running 폴더에서 conda activate Contest
pip install –r requirements.txt
python run_log.py --my_ai "rl" --opponent "random"
  -> Error 뜰 수 있는데 붉은 색이 아닌 하얀 색 Error이면 상관없이 다음꺼 진행
python evaluation_local.py --my_ai rl --opponent random --episode=100 --map=all
