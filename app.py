'''
설치 및 모듈 참조 
    - https://pypi.org/project/teachable-machine/
'''
from teachable_machine import TeachableMachine
import random 

# 모델 로딩
my_model = TeachableMachine(model_path = 'keras_model.h5', model_type='h5')
# 평가할 이미지 준비
imgs = ['choc1.jpg', 'choc2.jpg', 'choc3.jpg', 'choc4.jpg', 'chic1.jpg', 'chic2.jpg', 'chic3.jpg']
random.shuffle(imgs)
# 라벨
names = []
with open('labels.txt', 'rt') as f:
    names = f.readlines()

for img in imgs:
    res = my_model.classify_image(img)
    print(f"이미지 이름: {img}, 결과: {names[ res['highest_class_id'] ]}")