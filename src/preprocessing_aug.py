import os
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import random

# ===================
# 증강 함수 정의
# ===================
def random_augment(img):
    # 회전 (±15도)
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-15, 15), fillcolor=255)
    # 밝기 조정
    if random.random() < 0.3:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    # 이동
    if random.random() < 0.3:
        dx = int(random.uniform(-3, 3))
        dy = int(random.uniform(-3, 3))
        img = ImageOps.expand(img, border=(3, 3, 3, 3), fill=255)
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
        img = img.crop((3, 3, img.size[0]-3, img.size[1]-3))
    # 스케일(확대/축소)
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        w, h = img.size
        img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
        img = ImageOps.fit(img, (w, h), method=Image.BICUBIC, centering=(0.5, 0.5))
    # 노이즈
    if random.random() < 0.2:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 7, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    # 두께 변화 (dilation/erosion)
    if random.random() < 0.2:
        if random.random() < 0.5:
            img = img.filter(ImageFilter.MaxFilter(3))  # dilation
        else:
            img = img.filter(ImageFilter.MinFilter(3))  # erosion
    return img

# ===================
# 전처리/증강 메인 함수
# ===================
def preprocess_images(image_folder, target_size, has_label=True, augment=False, for_cnn=False):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))]
    image_data_list = []
    image_names = []
    labels = []

    for file_name in image_files:
        file_path = os.path.join(image_folder, file_name)
        try:
            img = Image.open(file_path).convert('L')
            img_resized = img.resize(target_size)
            
            # === 증강 적용 (옵션) ===
            if augment:
                img_resized = random_augment(img_resized)
            
            img_np = np.array(img_resized)
            
            if for_cnn:
                # CNN: (height, width, 1)로 변환
                img_3d = img_np.reshape(target_size[1], target_size[0], 1)  # (30, 100, 1)
                image_data_list.append(img_3d)
            else:
                # Dense: flatten
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

# ===================
# X, y 분리 함수
# ===================
def get_feature_label(df):
    X = np.array(df['image_data'].tolist())
    y = pd.get_dummies(df['label']).values
    return X, y
