# -*- coding: utf-8 -*-
import numpy as np #支援矩陣運算
import os,csv
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
from PIL import Image

train_csv="./aoi/train.csv"
train_path="./aoi/train_images"

# data_x(image) 與 data_y(label) 前處理
def img_preprocess(datapath):
    img_row,img_col=512,512 #定義圖片大小
    data_x=np.zeros((img_row,img_col)).reshape(1,img_row,img_col) #儲存圖片
    count=0 # 紀錄圖片張數
    # 讀取aoi 資料夾內的檔案
    for root,dirs,files in os.walk(datapath):
        for f in files:
            fullpath=os.path.join(root,f) # 取得檔案路徑
            img=Image.open(fullpath) # 開啟image 
            img=(np.array(img)/255).reshape(1,img_row,img_col) # 作正規化與reshape
            data_x=np.vstack((data_x,img))
            count+=1
    data_x=np.delete(data_x,[0],0) # 刪除np.zeros
    # 調整資料格式
    data_x=data_x.reshape(count,img_row,img_col,1)
    # 將label轉成one-hot.encoding
    return data_x

def csv_preprocess(datapath):
    data_y=[] #紀錄label
    num_class=6 # 種類6種
    with open(datapath, newline='') as csvfile:
        # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
        rows = csv.DictReader(csvfile)
        for row in rows:
            data_y.append(row['Label'])
    data_y=np_utils.to_categorical(data_y,num_class)
    return data_y

# 顯示訓練圖片
def show_train(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

data_x, data_y=img_preprocess(train_path),csv_preprocess(train_csv)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,test_size=0.2)  

model=Sequential() # 建立模型
# 第一層
model.add(Conv2D(filters=32,kernel_size=(2,2),padding='same',input_shape=(512,512,1),activation='relu')) # 建立卷基層
model.add(Dropout(0.25)) # Dropout隨機斷開輸入神經元，防止過度擬合，比例0.25
model.add(MaxPooling2D(pool_size=(2,2))) # 建立池化層
# 第二層
model.add(Conv2D(filters=64,kernel_size=(2,2),padding='same',activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
# 第三層
model.add(Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
# 第四層
model.add(Conv2D(filters=256,kernel_size=(2,2),padding='same',activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # 多維輸入一維化
model.add(Dropout(0.1))
model.add(Dense(1000,activation='relu')) 
model.add(Dropout(0.1))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=6,activation='softmax')) # 使用 softmax,將結果分類 units=10,10類
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy']) #損失函數、優化方法、成效衡量
train_history=model.fit(x_train,y_train,batch_size=32,epochs=100,verbose=1,validation_split=0.1)
show_train(train_history,'accuracy','val_accuracy')
show_train(train_history,'loss','val_loss')

# test_x,test_y=img_preprocess(test_path),csv_preprocess(test_csv)

score = model.evaluate(x_test,y_test, verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
model.save('cnn_model.h5')