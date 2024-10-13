import cv2
import os
from albumentations import (
    HorizontalFlip, RandomBrightnessContrast, Rotate, ShiftScaleRotate, 
    RandomGamma, Compose
)
import numpy as np

# 이미지 불러오기
image_dir = "/detect/image/"
left_image_path = os.path.join(image_dir, "left_1.png")
right_image_path = os.path.join(image_dir, "right_1.png")

left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)

# 증강 파이프라인 정의
augmentation_pipeline = Compose([
    HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    RandomBrightnessContrast(p=0.2),  # 밝기/대비 조정
    Rotate(limit=20, p=0.5),  # -20도 ~ 20도 사이로 회전
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),  # 이동/확대/회전
    RandomGamma(p=0.2),  # 감마 조정
])

# 증강 및 저장 함수
def augment_and_save(image, filename, num_augmentations=100):
    base_filename = filename.split('.')[0]  # 확장자 제거
    for i in range(num_augmentations):
        augmented = augmentation_pipeline(image=image)['image']
        augmented_filename = f"augmented_{base_filename}_{i+1}.png"
        augmented_path = os.path.join(image_dir, augmented_filename)
        cv2.imwrite(augmented_path, augmented)

# 원하는 증강 이미지 개수
num_augmentations = 100  # 예: 100개의 증강된 이미지를 생성

# left_1, right_1 이미지를 증강하여 복제
augment_and_save(left_image, "left_1.png", num_augmentations)
augment_and_save(right_image, "right_1.png", num_augmentations)

print(f"{num_augmentations}개의 증강된 이미지가 각각 생성되었습니다.")
