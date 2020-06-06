# -*- coding: utf-8 -*-
import numpy as np
import os
from tensorflow.keras import models
from PIL import Image

test_csv="./aoi/test.csv"
test_path="./aoi/test_images"

def img_preprocess(datapath):
    img_row,img_col=56,56 #定義圖片大小
    count=0 # 紀錄圖片張數
    data_x=np.zeros((img_row,img_col)).reshape(1,img_row,img_col) #儲存圖片
    # 讀取aoi 資料夾內的檔案
    for root,dirs,files in os.walk(datapath):
        for f in files:
            fullpath=os.path.join(root,f) # 取得檔案路徑
            img=img.resize((img_row,img_col),Image.NEAREST)
            img=Image.open(fullpath) # 開啟image 
            img=(np.array(img)/255).reshape(1,img_row,img_col) # 作正規化與reshape
            data_x=np.vstack((data_x,img))
            count+=1
    data_x=np.delete(data_x,[0],0) # 刪除np.zeros
    # 調整資料格式
    data_x=data_x.reshape(count,img_row,img_col,1)
    # 將label轉成one-hot.encoding
    return data_x,count

if os.path.isfile('cnn_model.h5'):
    model = models.load_model('cnn_model.h5')
else:
    print('No trained model found.')
    exit(-1)
    
data_x,totel=img_preprocess(test_path)
varification_code = list()
for i in range(totel):
    confidences = model.predict(np.array([data_x[i]]), verbose=0)
    result_class = model.predict_classes(np.array([data_x[i]]), verbose=0)
    varification_code.append(result_class[0])
    print('Digit {0}: Confidence=> {1}    Predict=> {2}'.format(i + 1, np.squeeze(confidences), np.squeeze(result_class)))
print('Predicted varification code:', varification_code)
with open(test_csv,'a+') as f :
    w=csv.writer(f)
    w.writecol(varification_code)
