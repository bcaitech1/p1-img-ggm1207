# AugMentation 정리

우선 사람이 보고 구별할 수 있어야 하며, 모델이 학습해야 하는 클래스(age, gender, mask)가 있을 때 편향이 생길 수 있는 정보들을 제거해주는 augmentation을 찾자.

**CoarseDropout**: ALL, 결정 경계
**ChannelShuffle**: ALL, 색깔 편향 제거
**ColorJitter**: ALL, 색깔 편향 제거
**Cutout**: ALL, 결정 경계
**FancyPCA**: ALL, 색깔 편향 제거 ( 자연스러움, 배경 색깔 )
**GridDistortion**: Mask, 손수건 빌런을 해결할 수 있지 않을까?
**GridDropout**: ALL, 결정 경계
**HorizontalFlip**: ALL, 데이터 증강
**HueSaturationValue**: ALL, 색깔 편향 제거
**RandomBrightnessContrast**: ALL, 밝기 편향 제거
**ToGray**: ALL, 색깔 편향 제거

