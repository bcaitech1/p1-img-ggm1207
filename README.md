# 1-Stage

**[WRAP UP REPORT](https://hackmd.io/@cdll-lo-ol-lo-ol/r1S5PKqHd)**

> WRAP UP 리포트에는 2주 간의 과정이 상세하게 적혀있습니다..! :smile:

## 문제 개요

마스크를 착용하는 건 COVID-19의 확산을 방지하는데 중요한 역할을 합니다. 제공되는 이 데이터셋은 사람이 마스크를 착용하였는지 판별하는 모델을 학습할 수 있게 해줍니다. 모든 데이터셋은 아시아인 남녀로 구성되어 있고 나이는 20대부터 70대까지 다양하게 분포하고 있습니다.

## 프로젝트 구조

### 모델 구조

![mag](https://i.imgur.com/GvtKJWR.png)

### 폴더 구조

```
- 1-STAGE
| (*.py, *.sh) - 모델 학습/추론에 사용된 모듈들입니다.
|----- notebook
|     (*.ipynb) - 모델 EDA 및 디버깅에 사용된 노트북들입니다.
|----- submissions
|     (*.csv)
|----- yaml_file
|     (*.yaml) - WandB Sweep 기능을 사용하는 설정 파일들입니다.
```

### 모듈 설명

- `config.py` : hyperparamter를 설정하는 모듈
- `network.py` : model이 구현된 모듈
- `prepare.py` : 데이터를 로드하는 모듈
- `train.py` : 모델 학습을 진행하는 모듈
- `predict.py` : 모델 성능을 측정하는 모듈
- `inference.py` : 모델 재학습 및 추론 결과를 제출하는 모듈 
- `metrics.py` : 로스 및 helper 함수들이 구현되어 있는 모듈
- `log_helper.py` : wandb 시각화 함수들이 구현되어 있는 모듈
- `inference.sh` : inference.py를 좀 더 쉽게 실행시켜주는 스크립트


### 학습 & 추론 파이프라인

![pipeline-01](https://i.imgur.com/T2RIv0h.png)

1. TRAIN
    - **하이퍼 파라미터 전략**(Grid, Bayes, Random)을 설정한 후 학습을 진행합니다.
2. HEURISTIC ( TRAIN )
    - **log, CM, GradCAMPP, Confidence**를 보면서 성능이 좋은 모델들을 찾아냅니다.
3. PREDICT
    - 제가 판단한 모델들을 조합해서 **total_valid_f1_score**가 가장 높은 모델의 조합(age, mask, gender)을 찾습니다.
4. HEURISTIC ( PREDICT )
    - PREDICT가 실행된 이후에 `wandb`에서 성능 좋은 모델의 조합을 쉽게 파악할 수 있습니다.
5. INFERENCE
    - 마지막으로는 가장 가능성이 높은 모델들을 **retrain** 하고 **submisson.csv**을 제출합니다.

### 노트북 설명

> 음 ... 제목이 곧 내용입니다.

1. baseline
2. data-eda
3. image-blending
4. log wandb images
5. predict fn
6. check predict
7. pr curve
8. predict model
9. check gradcam
10. data augmentation
11. log scores
12. age what false
13. baseline code
14. use only mask
15. loss test
16. advanced cv
17. coral-test
18. check precit
19. coral age visualization
20. ensemble
21. ktrain

---
