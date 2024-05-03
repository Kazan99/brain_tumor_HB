# ở đây chúng ta tést xem mô hình của chúng ta chạy thế nào 
# hoạt động ra sao

import cv2;
from keras.models import load_model;
from PIL import Image;
import numpy as np;

model = load_model('BrainTumor10EpochsCategorical.h5')

# model = load_model('BrainTumor10Epochs.h5');

image = cv2.imread('D:\\code\\luan_van_2\\test\\pred\\pred8.jpg');
# image = cv2.imread('D:\\code\\UNET_Segmentation\\dataset\\images\\13.png');

img = Image.fromarray(image);

img = img.resize((64, 64));

img = np.array(img);

input_img = np.expand_dims(img, axis=0);

#Phân loại nhị phân - BrainTumor10Epochs
# result = (model.predict(input_img) > 0.5).astype("int32");
# Phân loại nhiều lớp - BrainTumor10EpochsCategorical
percent = model.predict(input_img);
result = np.argmax(percent, axis= 1);
i = percent.argmax(axis=1)[0]

# if result == 1:
#     cv2.imshow('hien',img);
#     #wait for a key press to exit
#     cv2.waitKey(delay=0);

#     #clone All windows
#     cv2.destroyAllWindows();


print(percent.max()*100);
print(result);
print(i)