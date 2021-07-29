# YGL 2기 3팀 Final Project

```
파일 명 맨 앞에 test 혹은 test_가 붙은 파일은 테스트용 파일입니다.
```

## Topic
* YOLO를 활용한 hand gesture recognition과 패턴 비밀번호 인식 구현

## Team
* 전유상
* 박도현
* 정민형

---

## 실행 방법

#### 학습 데이터 처리 1단계

학습에 쓰일 영상을 이미지로 처리한다. 과정은 다음과 같다.

1. `config.py`를 실행하여 `hand_video/` 내에 라벨별로 빈 폴더를 생성한다.
2. 정해진 제스쳐를 취한 손을 찍은 영상을 `hand_video/`에 라벨별로 저장한다. (형식 : 00001.mp4와 같이 다섯자리 자연수)
3. `video2image.py`를 실행한다.

실행하면 다음과 같은 파일들이 생성된다.
* 폴더 `dataset/`, `dataset/origin/`, `dataset/origin_box/` 생성
* `dataset/origin/`에 영상에서 손이 인식된 **원본 이미지** 생성
* `dataset/origin_box/`에 원본 이미지에 손을 둘러싸는 box를 그린 **원본+box 이미지** 생성
* `dataset/`에 **원본+box 이미지**의 box 좌표를 저장한 텍스트 파일(`coordinates.csv`) 생성

#### 학습 데이터 처리 2단계

불량 데이터를 제거한다. 과정은 다음과 같다.

1. `dataset/origin_box/`에서 손에 box가 제대로 그려지지 않은 이미지를 직접 제거한다.
2. `refine_dataset.py`를 실행한다.

실행하면 다음과 같은 작업이 이루어진다.
* `dataset/origin_box/`에 존재하지 않으나, `dataset/origin/`에는 존재하는 이미지를 찾아내어 제거한다.
* 앞에서 제거한 이미지의 좌표 정보를 `dataset/coordinates.csv`에서 제거한다.

이와 같은 과정을 거쳐서 학습 데이터를 생성해낼 수 있다. 