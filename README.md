# 티처블머신과 파이썬을 사용한 인공지능 공부하기

## 시스템 요구사항
* Python >= 3.8  
* Microsoft Visual C++ Redistributable latest supported downloads  
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170  
  * 코드 실행시 msvcp.dll ~ 과 같은 에러가 발생할 경우 
  * 최신의 윈도우 런타임 배포판이 필요함. 
 
## 설치방법  
```bash
pip install teachable-machine
```

## 모듈 의존성  
```
numpy, Pillow, tensorflow
```  

## 티처블머신 모듈 사용 예시  
### 티처블머신에서 모델 만들기
1. https://teachablemachine.withgoogle.com 방문 모델만들기 
2. 모델 다운도르하기 Tensorflow 형으로 다운로드 하기 (keras_model.h5, lables.txt)
3. 압축파일 해제해서 파일 옮기기  

### 티처블머신 모듈과 파이썬으로 코드 작성하기

```py
from teachable_machine import TeachableMachine

my_model = TeachableMachine(model_path='keras_model.h5', model_type='h5')

img_path = 'images/my_image.jpg'

result = my_model.classify_image(img_path)

print('highest_class_id:', result['highest_class_id'])
print('all_predictions:', result['all_predictions'])
```

_highest_class_id_의 값은 'labels.txt'에서 읽어서 처리할 수 있음. 

### 티처블머신 모듈 사용하지 않고 파이썬과 텐서플로 사용하기 
```py
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('<IMAGE_PATH>').convert('RGB')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

print("Class: ", class_name)
print("Confidence Score: ", confidence_score)
```