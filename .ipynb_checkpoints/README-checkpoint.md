# 🧠 Sign Classification Project

손글씨 이미지를 분류하는 MLP 및 CNN 기반 모델 프로젝트입니다.  
총 9개의 실험 모델을 구현했으며, 단일 학습, K-Fold 교차 검증, 모델 추론 기능을 포함합니다.

---

## 폴더 구조

```
.
├── data/
│ ├── train/ # 학습 이미지 폴더
│ └── test/ # 테스트 이미지 폴더
├── models/ # 학습된 모델 저장 폴더 (.h5)
├── logs/ # 학습 로그 저장 폴더
├── outputs/ # 추론 결과 저장 폴더
├── src/
│ ├── preprocessing.py # 이미지 로드 및 전처리 함수
│ └── model.py # 실험 모델 정의 및 선택 딕셔너리
├── train_single.py # 단일 모델 학습 코드
├── train_kfold.py # K-Fold 교차 검증 학습 코드
└── inference.py # 저장된 모델로 추론 실행 코드
```

---

## 설치 및 실행

### 1. 학습 데이터 준비

* `./data/train` 및 `./data/test` 디렉토리에 `.png` 또는 `.jpg` 이미지 배치
* 파일명 예시: `sign_3_001.png` → 라벨은 파일명에서 자동 추출됨

### 2. 단일 모델 학습 실행

```bash
python train_single.py
```

### 3. K-Fold 교체 검정 실행

```bash
python train_kfold.py
```

### 4. 모델 추론 실행

```bash
python inference.py
```

---

## 실험 모델 목록 (9가지)

```python
from src.model import model_dict
model = model_dict['exp1'](input_dim=..., output_dim=...)
```

| 모델 키 | 구조 요약               |
| ---- | ------------------- |
| exp1 | 2048 + L2           |
| exp2 | 2048 → 1024         |
| exp3 | 1024 → 512          |
| exp4 | 128 + L2            |
| exp5 | 512 + Dropout → 256 |
| exp6 | 1024                |
| exp7 | 256 → 128 → 64      |
| exp8 | 2048 + Dropout      |
| exp9 | 300                 |

---

## 결과 파일

* `logs/train_single_log.csv`: 단일 학습 로그
* `logs/train_kfold_log.csv`: 교체 검정 정확도 로그
* `outputs/inference_result.csv`: 테스트 예측 결과 저장

---

## 사용 기술

* Python 3.10+
* TensorFlow 2.x
* scikit-learn
* pandas, numpy, PIL

---
## 📈 시각화 대시보드

- [`train_kfold_log.csv`](./logs/train_kfold_log.csv) 및 [`train_single_log.csv`](./logs/train_single_log.csv)를 기반으로 Tableau에서 시각화 가능
- [예시 대시보드 보기](https://public.tableau.com/app/profile/...) (← Tableau Public 링크)
- 주요 지표:
  - K-Fold 정확도 분포
  - 에폭별 정확도/손실 변화
  - 모델별 성능 비교

