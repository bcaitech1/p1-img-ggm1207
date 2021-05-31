# 1-Stage

[project-wrap-up-report](https://hackmd.io/@cdll-lo-ol-lo-ol/r1S5PKqHd)

## 모델 구조

![mag](https://i.imgur.com/GvtKJWR.png)

## 폴더 구조

```
- 1-STAGE
| (*.py, *.sh)
|----- notebook
|     (*.ipynb)
|----- submissions
|     (*.csv)
|----- yaml_file
|     (*.yaml)
```

---

## 모듈 설명

### Scripts (\*.py, \*.sh)

- config.py : hyperparamter를 설정하는 모듈
- network.py : model이 구현된 모듈
- prepare.py : 데이터를 로드하는 모듈
- train.py : 모델 학습을 진행하는 모듈
- predict.py : 모델 성능을 측정하는 모듈
- inference.py : 모델 재학습 및 추론 결과를 제출하는 모듈 
- metrics.py : 로스 및 helper 함수들이 구현되어 있는 모듈
- log_helper.py : wandb 시각화 함수들이 구현되어 있는 모듈
- inference.sh : inference.py를 좀 더 쉽게 실행시켜주는 스크립트

### Notebooks (\*.ipynb)

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
