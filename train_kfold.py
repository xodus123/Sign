import os
import pandas as pd
from sklearn.model_selection import train_test_split
# from src.preprocessing import preprocess_images, get_feature_label
from src.preprocessing_aug import preprocess_images, get_feature_label  # 증강 실험 시 주석만 바꿔서 사용
from src.model import model_dict, get_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === 설정 ===
model_name = 'exp9'
target_size = (100, 30)
epochs = 1000
batch_size = 16
mode = 'aug'  # 증강 실험 시 'aug'로만 바꿔주면 됨!

# === 데이터 로딩 (train 데이터만) ===
if model_name in ['exp8', 'exp9']:
    df = preprocess_images('./data/train', target_size, has_label=True, for_cnn=True)
else:
    df = preprocess_images('./data/train', target_size, has_label=True, for_cnn=False)

X, y = get_feature_label(df)

# === train/val 분할 (8:2) ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 모델 선택 및 생성 ===
if model_name in ['exp8', 'exp9']:
    model = get_model(model_name, X_train, y_train, input_shape=(30, 100, 1))
else:
    model = get_model(model_name, X_train, y_train)

# === 디렉토리 생성 ===
os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

# === 파일명 (실험구분까지 포함, 모델/로그) ===
model_path = f'./models/{model_name}_kfold_model_{mode}.h5'
log_path = './logs/train_kfold_log.csv'  # 실험구분 컬럼 있으니 한 파일에 누적!

checkpoint = ModelCheckpoint(
    model_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

# === 모델 학습 ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print(f"베스트 모델 저장 완료: {model_path}")

# === 베스트 val_accuracy, 그때의 val_loss, 에폭 구하기 ===
val_acc_list = history.history['val_accuracy']
val_loss_list = history.history['val_loss']
best_val_acc = max(val_acc_list)
best_epoch = val_acc_list.index(best_val_acc) + 1   # 1-based index
best_val_loss = val_loss_list[best_epoch - 1]
train_acc = history.history['accuracy'][-1]
train_loss = history.history['loss'][-1]

# === 실험 구분 컬럼 포함 로그 저장 (누적 CSV) ===
log_entry = pd.DataFrame([{
    'model': model_name,
    'mode': mode,                      # 실험구분 컬럼!
    'best_val_acc': best_val_acc,
    'best_val_loss': best_val_loss,
    'best_epoch': best_epoch,
    'train_acc': train_acc,
    'train_loss': train_loss
}])

if os.path.exists(log_path):
    prev_log = pd.read_csv(log_path)
    log_entry = pd.concat([prev_log, log_entry], ignore_index=True)

log_entry.to_csv(log_path, index=False)
print(f"단일 로그 저장 완료: {log_path}")
