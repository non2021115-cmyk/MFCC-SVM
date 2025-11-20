# MFCC-SVM

>>#how to run


<br>1. 컴퓨터에서 train.py 돌림 > 파일 몇개 획득
<br>2. run_real.py를 실시간으로 data/real/ 폴더를 읽어 새로 파일이 생성되면 그걸 판단하는 실행 파일로 변환 시키기
<br>3. 라스베리파이에서 몇초마다 음성 데이터를 받고 그걸 data/real/에 저장하는 파이썬파일 만들기
<br>4. 실제 하드웨어(라스베리 파이, 정확히 어떤 식으로 작동하는지는 모름)에 train.py를 돌려 받은 파일 몇개랑 run_real.py, data/real/, 그리고 3번에서 만든 파일 넣기
<br>5. 라스베리파이에서 run_real.py 실행하기
