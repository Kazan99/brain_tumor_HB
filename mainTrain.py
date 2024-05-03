#Распознавании опухолей головного мозга с использованием методов глубокого обучения
# Разработка модуля диагностики опухолей головного мозга с использованием методов глубокого обучения - тема для ВКР
# Tạo quy trình đào tạo mô hình thực tế với khối u não (model cnn)
# Xây dựng mô hình bằng cnn và có một băng chuyền tensorflow

import cv2;
import os;
import pandas as pd;
import tensorflow as tf;
import matplotlib.pyplot as plt;
from tensorflow import keras;
from PIL import Image; # Предоставляет поддержку при открытии, управлении и сохранении многих форматов изображения
import numpy as np;
from sklearn.model_selection import train_test_split;
# sử dụng thư viện cho mảng thần kinh nhân tạo 
# keras - позволяет быстрее создавать и настраивать модели — схемы, по которым распространяется и подсчитывается информация при обучении.
# Но сложных математических вычислений Keras не выполняет и используется как надстройка над другими библиотеками.
from keras.utils import normalize;
from keras.models import Sequential;
from keras.layers import Conv2D, MaxPooling2D;
from keras.layers import Activation, Dropout, Flatten, Dense;
from keras.layers import LeakyReLU;
from keras.utils import to_categorical;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import classification_report;

#khai bào đường dẫn chứa các đataset để 
image_directory = 'brain_tumor_dataset/';
no_tumor_images = os.listdir(image_directory + 'no/');
yes_tumor_images = os.listdir(image_directory + 'yes/');
dataset = [];
label = [];

INPUT_SIZE = 64;

# print(no_tumor_images);

# path = '1 no.jpeg';

# print(path.split('.')[1])

# đọc các file ảnh để cho hoc máy 
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1] == 'JPG' or image_name.split('.')[1] == 'jpg' or image_name.split('.')[1] == 'png' or image_name.split('.')[1] == 'jpeg'): # Функция split сканирует всю строку и разделяет ее в случае нахождения разделителя
        image = cv2.imread(image_directory + 'no/' + image_name); #imread - читать изображение
        image = Image.fromarray(image, 'RGB');
        image =image.resize((INPUT_SIZE, INPUT_SIZE)); #делать изображение в размер 64*64
        dataset.append(np.array(image));
        label.append(0);

for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1] == 'JPG' or image_name.split('.')[1] == 'jpg' or image_name.split('.')[1] == 'png' or image_name.split('.')[1] == 'jpeg'): # Функция split сканирует всю строку и разделяет ее в случае нахождения разделителя
        image = cv2.imread(image_directory + 'yes/' + image_name); #imread - читать изображение
        image = Image.fromarray(image, 'RGB');
        image =image.resize((INPUT_SIZE, INPUT_SIZE)); #делать изображение в размер 64*64
        dataset.append(np.array(image));
        label.append(1);

#hiển thị số lượng ảnh trong 2 file (NO , Yes) nhằm mục đích là để biết ta đã truy cập đuọc tất cả các file chưa
print(len(dataset)); 
print(len(label)); 

dataset = np.array(dataset); 
label = np.array(label);
print(len(dataset)); 
print(len(label)); 

# df= pd.DataFrame(dataset);
# df['Target'] = label;
# df.shape;
# #input data
# x = df.iloc[:,:-1];
# y = df.iloc[:,-1];

x_train,x_test, y_train, y_test = train_test_split(dataset, label, test_size= 0.2, random_state= 77);

# Reshape = (n, image_width,image_height, n_channel);

# print(x_train.shape);
# print(y_train.shape);

# print(x_test.shape);
# print(y_test.shape);

x_train = normalize(x_train, axis= 1);
x_test = normalize(x_test, axis= 1);

y_train = to_categorical(y_train, num_classes= 2);
y_test = to_categorical(y_test, num_classes= 2);

# Model Buiding 
# 64, 64, 3

model = Sequential(); # cấu trúc CNN

#trọng số "he-uniform"
#tạo bộ đệm để đầu ra giống đầu vào
#hạn chế mất mát dữ liệu khi tích chập có giá trị âm
#pooling các giá trị lowns nhất trong vùng 2x2
#9 lớp convolutional 3x3
#áp dụng các bộ lọc (3x3) để trích xuất các đặc trưng từ đầu vào 2D
model.add(Conv2D(32, (3,3), input_shape = (INPUT_SIZE, INPUT_SIZE, 3)));
model.add(Activation('relu'));
model.add(LeakyReLU(alpha = 0.1)); #các giá trị âm nhân với 0.1 thay vì cho bằng 0 dể tránh mất mát dưk liệu
model.add(MaxPooling2D(pool_size= (2,2)));

model.add(Conv2D(32, (3,3), kernel_initializer= 'he_uniform'));
model.add(Activation('relu'));
model.add(LeakyReLU(alpha = 0.1));
model.add(MaxPooling2D(pool_size= (2,2)));
# giảm kich thước cảu feature map
#feature map sẽ được chia thành các vùng 2x2 và lấy giá trị lớn nhất trong mỗi vùng
#sẽ được chọn để tạo thành feature map mới với kích thước giảm đi một nửa theo chiều ngang và dọc
model.add(Conv2D(64, (3,3), kernel_initializer= 'he_uniform'));
model.add(Activation('relu'));
model.add(LeakyReLU(alpha = 0.1));
model.add(MaxPooling2D(pool_size= (2,2)));

model.add(Conv2D(128, (3,3), kernel_initializer= 'he_uniform'));
model.add(Activation('relu'));
model.add(LeakyReLU(alpha = 0.1));
model.add(MaxPooling2D(pool_size= (2,2)));


model.add(Flatten()); #chuyển đầu ra 2D thành 1D để đưa vào lớp fullconnected
model.add(Dense(64));
model.add(Activation('relu'));
model.add(Dropout(0.5)); # loại bỏ bớt 1 số thành phần để tránh ovefitting không tổng quát tốt được quá trình học
model.add(Dense(2)); # fullconnected 2 ngõ ra tương ứng vói yes và no
# model.add(Activation('sigmoid'));
model.add(Activation('softmax')); # đưa ra xác xuất dự đoán cho mỗi lớp

model.summary();

# Binary CrossEntropy = 1, sigmoid
# Categorical Cross Entryopy= 2, softmax

# model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy']); 
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy']);

training_model = model.fit(x_train, y_train, 
        batch_size= 16, 
        verbose= 1,steps_per_epoch= 150, epochs= 20, 
        validation_data= (x_test, y_test),
        shuffle= False);

# model.save('BrainTumor10Epochs.h5');
model.save('BrainTumor10EpochsCategorical.h5');
# model.save('BrainTumor10EpochsCategorical_1.h5');

plt.plot(training_model.history["loss"], label = "Train loss");
plt.plot(training_model.history["val_loss"], label = "val loss");
plt.legend();
plt.title ("График потерь (Loss)");
plt.xlabel("Epochs");
plt.ylabel("Loss value");
plt.show();
plt.plot(training_model.history["accuracy"], label = "Train Accuracy");
plt.plot(training_model.history["val_accuracy"], label = "val acc");
plt.legend();
plt.title ("График точность (Accuracy)");
plt.xlabel("Epochs");
plt.ylabel("Accuracy value");
plt.show();