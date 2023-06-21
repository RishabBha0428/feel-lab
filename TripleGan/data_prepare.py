import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os.path, sys
import os
import pandas as pd
from zipfile import ZipFile
from tqdm import tqdm
from PIL import Image
import os
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_data(img_folder):
  img_data_array=[]
  class_name=[]
  def imagetensor(imagedir):
    img = tf.keras.preprocessing.image.load_img(imagedir, color_mode='rgb', target_size = (32,32))
    image_array = tf.keras.preprocessing.image.img_to_array(img)
    return image_array 

  for dir1 in os.listdir(img_folder):
      for file in os.listdir(os.path.join(img_folder, dir1)):
          image_path= os.path.join(img_folder, dir1,  file)
          image= imagetensor(image_path)
          
          img_data_array.append(image)
          class_name.append(dir1)

          
  target_dict={k: v for v, k in enumerate(np.unique(class_name))}
  target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
  target_val = list(map(int,target_val))

  images = np.array(img_data_array)
  labels = np.array(target_val)
  train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.20, random_state=42)
  print("DATA LOADED")
 
  return (train_images, train_labels), (test_images, test_labels)
  
