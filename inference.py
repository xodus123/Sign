import os
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_images, get_feature_label
from tensorflow.keras.models import load_model

# === 모델명/경로 설정 ===
model_name = 'exp9'
mode = 'aug' # basic / aug 증강
train_type = 'kfold' # single / kfold
# model_path = f'./models/{model_name}_model_{mode}.h5'
model_path = f'./models/{model_name}_kfold_model_{mode}.h5' # kfold 실행

# === 모델 불러오기 ===
model = load_model(model_path)
 
# === 테스트 데이터 로딩 ===
has_label = True  # 또는 False (정답 비교 안 할 때)
for_cnn = (model_name in ['exp8', 'exp9'])

test_df = preprocess_images(
    './data/test',
    target_size=(100, 30),
    has_label=has_label,
    for_cnn=for_cnn
)

if has_label and 'label' in test_df.columns:
    X_test, y_test = get_feature_label(test_df)
else:
    X_test = np.array(test_df['image_data'].tolist())
    y_test = None

# === 예측 및 정확도 계산 ===
pred_probs = model.predict(X_test)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(pred_classes == true_classes)

# === 정확도 누적 저장 ===
os.makedirs('./outputs', exist_ok=True)
acc_log_path = './outputs/inference_accuracy.csv'

acc_entry = pd.DataFrame([{
    'model': model_name,
    'mode' : mode,
    'train_type': train_type,
    'accuracy': accuracy
}])

# 이전 결과와 누적 저장
if os.path.exists(acc_log_path):
    prev = pd.read_csv(acc_log_path)
    acc_entry = pd.concat([prev, acc_entry], ignore_index=True)

acc_entry.to_csv(acc_log_path, index=False)
print(f"✅ accuracy 누적 저장 완료: {acc_log_path} (정확도={accuracy:.4f})")
