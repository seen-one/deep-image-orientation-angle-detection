from utils import rotate_preserve_size
import glob
import os
import numpy as np
import cv2
import random

import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.utils import Sequence

from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

class ViTRotGenerator(Sequence):
    def __init__(self, image_dir, batch_size, dim):
        self.files = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.batch_size = batch_size
        self.dim = dim
        
    def __len__(self):
        if len(self.files) % self.batch_size == 0:
            return len(self.files) // self.batch_size
        return len(self.files) // self.batch_size + 1
    
    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_files = self.files[batch_slice]

        X_conv = []
        X_vit = []
        y = []
        
        for i, f in enumerate(batch_files):
            try:
                angle = float(np.random.choice(range(0, 360)))
                img = rotate_preserve_size(f, angle, (self.dim, self.dim))
                img = np.array(img)
                X_vit.append(img)

                img = np.expand_dims(img, axis=0)
                X_conv.append(img)
                y.append(angle)

            except:
                pass
        
        X_vit = feature_extractor(images=X_vit, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)
        X_conv = np.concatenate(X_conv, axis=0)
        y = np.array(y)

        return [X_vit, X_conv], y
    
    def on_epoch_end(self):
        random.shuffle(self.files)


class ViTValidationTestGenerator(Sequence):
    def __init__(self, image_dir, df_label_path, batch_size, dim, mode):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.dim = dim
        self.mode = mode
        
        df_label = pd.read_csv(df_label_path)
        self.df = df_label[df_label["mode"] == self.mode].reset_index(drop=True)
        
    def __len__(self):
        total = self.df.shape[0]
        if total % self.batch_size == 0:
            return total // self.batch_size
        return total // self.batch_size + 1
    
    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        df_batch = self.df[batch_slice].reset_index(drop=True).copy()
        

        X_conv = []
        X_vit = []
        y = []
        
        for i in range(len(df_batch)):
            try:
                angle = df_batch.angle[i]
                path = os.path.join(self.image_dir, df_batch.image[i])
                img = rotate_preserve_size(path, angle, (self.dim, self.dim))

                img = np.array(img)
                X_vit.append(img)

                img = np.expand_dims(img, axis=0)
                X_conv.append(img)
                y.append(angle)

            except:
                pass
        
        X_vit = feature_extractor(images=X_vit, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)
        X_conv = np.concatenate(X_conv, axis=0)
        y = np.array(y)

        return [X_vit, X_conv], y


class RotGenerator(Sequence):
    def __init__(self, image_dir, batch_size, dim, channels_first=False, is_vit=False):
        self.files = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.batch_size = batch_size
        self.dim = dim
        self.channels_first = channels_first
        self.is_vit = is_vit
        
    def __len__(self):
        if len(self.files) % self.batch_size == 0:
            return len(self.files) // self.batch_size
        return len(self.files) // self.batch_size + 1
    
    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_files = self.files[batch_slice]
        
        # X = np.zeros(shape=(len(batch_files), self.dim, self.dim, 3))
        # y = np.zeros(shape=(len(batch_files), ))

        X = []
        y = []
        
        for i, f in enumerate(batch_files):
            try:
                angle = float(np.random.choice(range(0, 360)))
                img = rotate_preserve_size(f, angle, (self.dim, self.dim))
                img = np.array(img)
                if self.is_vit:
                    X.append(img)
                else:
                    if self.channels_first:
                        img = img.transpose(2, 0, 1)

                    img = np.expand_dims(img, axis=0)
                    X.append(img)
                    # X[i] = img
                    # y[i] = angle
                y.append(angle)

            except:
                pass
        
        if self.is_vit:
            X = feature_extractor(images=X, return_tensors="tf")["pixel_values"]
            X = np.array(X)
        else:
            X = np.concatenate(X, axis=0)
        y = np.array(y)

        return X, y
    
    def on_epoch_end(self):
        random.shuffle(self.files)


# In[83]:


class ValidationTestGenerator(Sequence):
    def __init__(self, image_dir, df_label_path, batch_size, dim, mode, channels_first=False, is_vit=False):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.dim = dim
        self.mode = mode
        self.channels_first = channels_first
        self.is_vit = is_vit
        
        df_label = pd.read_csv(df_label_path)
        df_label["angle"] = df_label["angle"].astype("float")
        self.df = df_label[df_label["mode"] == self.mode].reset_index(drop=True)
        
    def __len__(self):
        total = self.df.shape[0]
        if total % self.batch_size == 0:
            return total // self.batch_size
        return total // self.batch_size + 1
    
    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        df_batch = self.df[batch_slice].reset_index(drop=True).copy()
        
        # X = np.zeros(shape=(len(df_batch), self.dim, self.dim, 3))
        # y = np.zeros(shape=(len(df_batch), ))

        X = []
        y = []
        
        for i in range(len(df_batch)):
            try:
                angle = df_batch.angle[i]
                path = os.path.join(self.image_dir, df_batch.image[i])
                img = rotate_preserve_size(path, angle, (self.dim, self.dim))

                img = np.array(img)
                if self.is_vit:
                    X.append(img)
                else:
                    if self.channels_first:
                        img = img.transpose(2, 0, 1)

                    img = np.expand_dims(img, axis=0)
                    X.append(img)
                    # X[i] = img
                    # y[i] = angle
                y.append(angle)

            except:
                pass
        
        if self.is_vit:
            X = feature_extractor(images=X, return_tensors="tf")["pixel_values"]
            X = np.array(X)
        else:
            X = np.concatenate(X, axis=0)
        y = np.array(y)

        return X, y