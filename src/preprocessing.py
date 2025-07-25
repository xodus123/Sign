import os
import pandas as pd
from PIL import Image
import numpy as np

# 이미지 사이즈 설정 (width=100, height=30)
target_size = (100, 30)  # (width, height)

# 이미지 전처리 함수 (Dense/MLP + CNN 모두 지원)
def preprocess_images(image_folder, target_size, has_label=True, for_cnn=False):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))]
    image_data_list = []
    image_names = []
    labels = []

    for file_name in image_files:
        file_path = os.path.join(image_folder, file_name)
        try:
            img = Image.open(file_path).convert('L')  # 흑백 변환
            img_resized = img.resize(target_size)     # 리사이즈 (width, height)
            img_np = np.array(img_resized)

            if for_cnn:
                # CNN: (height, width, 1)로 변환 (Keras 기준)
                img_3d = img_np.reshape(target_size[1], target_size[0], 1)  # (30, 100, 1)
                image_data_list.append(img_3d)
            else:
                # Dense: flatten (벡터)
                img_1d = img_np.flatten()
                image_data_list.append(img_1d)

            image_names.append(file_name)

            if has_label:
                try:
                    label = int(file_name.split('_')[1])
                except (IndexError, ValueError):
                    label = None
                labels.append(label)
        except Exception as e:
            print(f"Error processing image {file_name}: {e}")
            continue

    if has_label:
        return pd.DataFrame({'image_data': image_data_list, 'label': labels}, index=image_names)
    else:
        return pd.DataFrame({'image_data': image_data_list}, index=image_names)

# X, y 분리 함수 (One-hot 인코딩 포함)
def get_feature_label(df):
    X = np.array(df['image_data'].tolist())
    y = pd.get_dummies(df['label']).values
    return X, y
