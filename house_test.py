
import os
import cv2
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras_retinanet.models import load_model
import sys
folder_path = sys.argv[1]
dst_folder_path = sys.argv[2]
model_path='./data/model/'
def nms(boxes,scores):
    box,score=[],[]
    detetion_index=np.where(scores!=-1)[0]
    for i in detetion_index:
        box.append(boxes[i].tolist())
        score.append(scores[i].tolist())  
    box=np.array(box)
    box_index, score = tf.image.non_max_suppression_with_scores(
      box, score, 99, iou_threshold=0.2, score_threshold=0.7) 
    return box[box_index.numpy()], score.numpy()

model=load_model(model_path+'H_whole_model.h5')


file_list=os.listdir(folder_path)
src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
house_object=np.array(['house','window','roof','wall','door','chimney'])
dst_list=[]
dst_box=[]
dst_label=[]
dst_score=[]
for file_name in file_list:
    try:
        input=cv2.imread(folder_path+file_name)
        input=cv2.resize(input,(1024,1024))
        input=np.reshape(input,(1,1024,1024,3))
        src_img=np.concatenate((src_img,input),axis=0)
        dst_list.append(file_name)
    except:
        print(file_name+'은 이미지 파일이 아닙니다')

boxes, scores, labels=model.predict(src_img)
detetion_index=np.where(scores!=-1)[0]
prediction_box,prediction_label,prediction_score=[],[],[]

for i in range(len(dst_list)):
    class_label=np.unique(labels[i])
    class_label=np.delete(class_label,np.where(class_label==-1))
    dst_box=[]
    dst_label=[]
    dst_score=[]
    for j in class_label:
        class_index=np.where(labels[i]==j)[0]
        src_box,src_score=nms(boxes[i][class_index], scores[i][class_index])
        dst_box.append(src_box)
        dst_score.append(src_score)
        dst_label.append(j)
    prediction_box.append(dst_box)
    prediction_score.append(dst_score)
    prediction_label.append(dst_label)

fontFace = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.5
thickness = 1
try:
    os.mkdir(dst_folder_path+'csv')
except:
    pass
for i in range(len(dst_list)):
    input=cv2.imread(folder_path+dst_list[i])
    input=cv2.resize(input,(1024,1024))
    img=np.copy(input)
    csv_data = {'x1': [],
            'y1': [],
            'x2': [],
            'y2': [],
            'class_name': [],
            'score':[]}
    for j in range(len(prediction_box[i])):
        for k in range(len(prediction_box[i][j])):
            pt1=(int(prediction_box[i][j][k][0]),int(prediction_box[i][j][k][1]))
            pt2=(int(prediction_box[i][j][k][2]),int(prediction_box[i][j][k][3]))
            img=cv2.rectangle(img,pt1,pt2,(0,255,0),4)
            label =house_object[prediction_label[i][j]] + ' score : '+str(int(prediction_score[i][j][k]*100)/100)
            csv_data['x1'].append(int(prediction_box[i][j][k][0]))
            csv_data['y1'].append(int(prediction_box[i][j][k][1]))
            csv_data['x2'].append(int(prediction_box[i][j][k][2]))
            csv_data['y2'].append(int(prediction_box[i][j][k][3]))
            csv_data['class_name'].append(house_object[prediction_label[i][j]])
            csv_data['score'].append(prediction_score[i][j][k])
            labelSize = cv2.getTextSize(label, fontFace, fontScale, thickness)
            cv2.rectangle(img, pt1, (pt1[0]+labelSize[0][0],pt1[1]-labelSize[0][1]), (0,255,0), cv2.FILLED)
            cv2.putText(img, label, pt1, fontFace, fontScale, (0,0,0), thickness)
    all_data = pd.DataFrame(csv_data)
    all_data.to_csv(dst_folder_path+'csv/'+dst_list[i][0:dst_list[i].find('.')]+'.csv',index=False,encoding='cp949',header=False)
    cv2.imwrite(dst_folder_path+dst_list[i],img)
