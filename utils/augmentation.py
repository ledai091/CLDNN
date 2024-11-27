from glob import glob
import cv2
import shutil
import re
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Callable

class ImageAugmenter:
    def __init__(self):
        self.augmentation_methods: Dict[int, Callable] = {
            0: self._flip_horizontal,
            1: self._rotate_90_clockwise,
            2: self._add_gaussian_noise,
            3: self._adjust_brightness,
            4: self._apply_gaussian_blur,
            5: self._convert_grayscale,
            6: self._adjust_contrast,
            7: self._adjust_saturation,
            8: self._apply_cartoon_effect,
            9: self._adjust_gamma,
            10: self._apply_color_overlay
        }
    
    def augment_dataset(self, 
                       X_train: pd.Series, 
                       y_train: pd.Series, 
                       num_augmentations: int = 5) -> Tuple[pd.Series, pd.Series]:

        list_idx = [i for i, j in enumerate(y_train) if j == 0]
        path_imbalanced = [X_train.iloc[i] for i in list_idx]
        
        for path in path_imbalanced:
            img_paths = sorted(glob(os.path.join(path, '*.jpg')), 
                             key=self._custom_sort_key)
            
            all_selects = np.arange(len(self.augmentation_methods))
            np.random.shuffle(all_selects)
            used_methods = all_selects[:num_augmentations]
            
            for method_id in used_methods:
                dir = f'{path}_augmentation_{method_id}'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                

                for img_path in img_paths:
                    img_augmented = self.augment_image(img_path, method_id)
                    file_name = os.path.splitext(os.path.basename(img_path))[0] + '_augmented'
                    cv2.imwrite(f'{dir}/{file_name}.jpg', img_augmented)
                

                X_train = pd.concat([X_train, pd.Series([dir])], ignore_index=True)
                y_train = pd.concat([y_train, pd.Series([0])], ignore_index=True)
        
        return X_train, y_train
    
    def augment_image(self, img_path: str, method_id: int) -> np.ndarray:
        img = cv2.imread(img_path)
        if method_id in self.augmentation_methods:
            return self.augmentation_methods[method_id](img)
        return img
    
    @staticmethod
    def delete_augmentations(base_path: str = 'img_resize/*/**') -> None:
        folder_path = glob(base_path)
        for path in folder_path:
            if 'augmentation' in path:
                shutil.rmtree(path)
                print(f'Deleted folder: {path}')
    
    @staticmethod
    def _custom_sort_key(filename: str) -> int:
        match = re.search(r'frame_(\d+)', filename)
        return int(match.group(1)) if match else filename
    @staticmethod
    def _flip_horizontal(img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 1)
    
    @staticmethod
    def _rotate_90_clockwise(img: np.ndarray) -> np.ndarray:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    @staticmethod
    def _add_gaussian_noise(img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    
    @staticmethod
    def _adjust_brightness(img: np.ndarray, brightness: int = 50) -> np.ndarray:
        return cv2.add(img, (brightness, brightness, brightness, 0))
    
    @staticmethod
    def _apply_gaussian_blur(img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, (5, 5), 0)
    
    @staticmethod
    def _convert_grayscale(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def _adjust_contrast(img: np.ndarray, contrast: float = 1.5) -> np.ndarray:
        return cv2.convertScaleAbs(img, alpha=contrast, beta=0)
    
    @staticmethod
    def _adjust_saturation(img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 1.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def _apply_cartoon_effect(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)
    
    @staticmethod
    def _adjust_gamma(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    @staticmethod
    def _apply_color_overlay(img: np.ndarray) -> np.ndarray:
        overlay_color = np.random.randint(0, 256, 3).tolist()
        overlay = np.full(img.shape, overlay_color, dtype=np.uint8)
        return cv2.addWeighted(img, 0.8, overlay, 0.2, 0)