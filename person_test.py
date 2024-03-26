
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
def nms(boxes,scores,count,threshold):
    box,score=[],[]
    detetion_index=np.where(scores!=-1)[0]
    for i in detetion_index:
        box.append(boxes[i].tolist())
        score.append(scores[i].tolist())  
    box=np.array(box)
    box_index, score = tf.image.non_max_suppression_with_scores(
      box, score, count, iou_threshold=0.2, score_threshold=threshold) 
    return box[box_index.numpy()], score.numpy()
model=load_model(model_path+'P_whole_model.h5')


file_list=os.listdir(folder_path)
src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
person_object=np.array(['person','head','face','eye','nose','mouth','ear','arm','hand','leg','foot','upper','lower'])
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
        if j==12:
            src_box,src_score=nms(boxes[i][class_index], scores[i][class_index],1,0.7)
        elif j==11:
            src_box,src_score=nms(boxes[i][class_index], scores[i][class_index],1,0.7)
        elif j==1:
            src_box,src_score=nms(boxes[i][class_index], scores[i][class_index],1,0.7)
        elif j==2:
            src_box,src_score=nms(boxes[i][class_index], scores[i][class_index],1,0.7)
        dst_box.append(src_box)
        dst_score.append(src_score)
        dst_label.append(j)
    prediction_box.append(dst_box)
    prediction_score.append(dst_score)
    prediction_label.append(dst_label)

face_src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
face_src_index=[]
face_ratio_w=[]
face_ratio_h=[]
x1,x2,y1,y2=[],[],[],[]
for i in range(len(dst_list)):
    try:
        face_index=prediction_label[i].index(2)
        if len(prediction_box[i][face_index])==0: 
            continue
        face_box=prediction_box[i][face_index][0]
        x1.append(face_box[0])
        y1.append(face_box[1])
        x2.append(face_box[2])
        y2.append(face_box[3])
        
        face_ratio_w.append((face_box[2]-face_box[0])/1024)
        face_ratio_h.append((face_box[3]-face_box[1])/1024)
        face_img = src_img[i][int(face_box[1]): int(face_box[3]), int(face_box[0]): int(face_box[2])]
        face_img=cv2.resize(face_img,(1024,1024))
        face_img=np.reshape(face_img,(1,1024,1024,3))
        face_src_img=np.concatenate((face_src_img,face_img),axis=0)
        face_src_index.append(i)
    except:
        continue
model=load_model(model_path+'P_face_model.h5')
face_boxes, face_scores, face_labels=model.predict(face_src_img)  

for i in range(len(face_src_index)):
    class_label=np.unique(face_labels[i])
    class_label=np.delete(class_label,np.where(class_label==-1))
    dst_box=[]
    dst_label=[]
    dst_score=[]
    for j in class_label:
        class_index=np.where(face_labels[i]==j)[0]
        if j==3:
            src_box,src_score=nms(face_boxes[i][class_index], face_scores[i][class_index],2,0.3)
        elif j==4:
            src_box,src_score=nms(face_boxes[i][class_index], face_scores[i][class_index],1,0.3)
        else:
            src_box,src_score=nms(face_boxes[i][class_index], face_scores[i][class_index],1,0.3)
        src_box[:,0]=(src_box[:,0])*face_ratio_w[i]+x1[i]
        src_box[:,1]=(src_box[:,1])*face_ratio_h[i]+y1[i]
        src_box[:,2]=(src_box[:,2])*face_ratio_w[i]+x1[i]
        src_box[:,3]=(src_box[:,3])*face_ratio_h[i]+y1[i]
        prediction_box[face_src_index[i]].append(src_box)
        prediction_score[face_src_index[i]].append(src_score)
        prediction_label[face_src_index[i]].append(j)
        
lower_src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
lower_src_index=[]
lower_ratio_w=[]
lower_ratio_h=[]
x1,x2,y1,y2=[],[],[],[]
for i in range(len(dst_list)):
    try:
        lower_index=prediction_label[i].index(12)
        if len(prediction_box[i][lower_index])==0: 
            continue
        lower_box=prediction_box[i][lower_index][0]
        x1.append(lower_box[0])
        y1.append(lower_box[1])
        if (9*lower_box[2]/8)>1024:
            x2.append(1024)
        else:
            x2.append(9*lower_box[2]/8)
        if (9*lower_box[3]/8)>1024:
            y2.append(1024)
        else:
            y2.append(9*lower_box[3]/8)
        
        lower_ratio_w.append((x2[-1]-x1[-1])/1024)
        lower_ratio_h.append((y2[-1]-y1[-1])/1024)
        lower_img = src_img[i][int(y1[-1]): int(y2[-1]), int(x1[-1]): int(x2[-1])]
        lower_img=cv2.resize(lower_img,(1024,1024))
        lower_img=np.reshape(lower_img,(1,1024,1024,3))
        lower_src_img=np.concatenate((lower_src_img,lower_img),axis=0)
        lower_src_index.append(i)
    except:
        continue
model=load_model(model_path+'P_lower_model.h5')
lower_boxes, lower_scores, lower_labels=model.predict(lower_src_img)  

for i in range(len(lower_src_index)):
    class_label=np.unique(lower_labels[i])
    class_label=np.delete(class_label,np.where(class_label==-1))
    dst_box=[]
    dst_label=[]
    dst_score=[]
    for j in class_label:
        class_index=np.where(lower_labels[i]==j)[0]
        if j==9:
            src_box,src_score=nms(lower_boxes[i][class_index], lower_scores[i][class_index],2,0.5)
        elif j==10:
            src_box,src_score=nms(lower_boxes[i][class_index], lower_scores[i][class_index],2,0.5)
        src_box[:,0]=(src_box[:,0])*lower_ratio_w[i]+x1[i]
        src_box[:,1]=(src_box[:,1])*lower_ratio_h[i]+y1[i]
        src_box[:,2]=(src_box[:,2])*lower_ratio_w[i]+x1[i]
        src_box[:,3]=(src_box[:,3])*lower_ratio_h[i]+y1[i]
        prediction_box[lower_src_index[i]].append(src_box)
        prediction_score[lower_src_index[i]].append(src_score)
        prediction_label[lower_src_index[i]].append(j)

upper_src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
upper_src_index=[]
upper_ratio_w=[]
upper_ratio_h=[]
x1,x2,y1,y2=[],[],[],[]
for i in range(len(dst_list)):
    try:
        upper_index=prediction_label[i].index(11)
        if len(prediction_box[i][upper_index])==0: 
            continue
        upper_box=prediction_box[i][upper_index][0]
        x1.append(upper_box[0])
        y1.append(upper_box[1])
        if (9*upper_box[2]/8)>1024:
            x2.append(1024)
        else:
            x2.append(9*upper_box[2]/8)
        if (9*upper_box[3]/8)>1024:
            y2.append(1024)
        else:
            y2.append(9*upper_box[3]/8)
        
        upper_ratio_w.append((x2[-1]-x1[-1])/1024)
        upper_ratio_h.append((y2[-1]-y1[-1])/1024)
        upper_img = src_img[i][int(y1[-1]): int(y2[-1]), int(x1[-1]): int(x2[-1])]
        upper_img=cv2.resize(upper_img,(1024,1024))
        upper_img=np.reshape(upper_img,(1,1024,1024,3))
        upper_src_img=np.concatenate((upper_src_img,upper_img),axis=0)
        upper_src_index.append(i)
    except:
        continue
model=load_model(model_path+'P_upper_model.h5')
upper_boxes, upper_scores, upper_labels=model.predict(upper_src_img)  

for i in range(len(upper_src_index)):
    class_label=np.unique(upper_labels[i])
    class_label=np.delete(class_label,np.where(class_label==-1))
    dst_box=[]
    dst_label=[]
    dst_score=[]
    for j in class_label:
        class_index=np.where(upper_labels[i]==j)[0]
        if j==7:
            src_box,src_score=nms(upper_boxes[i][class_index], upper_scores[i][class_index],2,0.5)
        elif j==8:
            src_box,src_score=nms(upper_boxes[i][class_index], upper_scores[i][class_index],2,0.5)
        src_box[:,0]=(src_box[:,0])*upper_ratio_w[i]+x1[i]
        src_box[:,1]=(src_box[:,1])*upper_ratio_h[i]+y1[i]
        src_box[:,2]=(src_box[:,2])*upper_ratio_w[i]+x1[i]
        src_box[:,3]=(src_box[:,3])*upper_ratio_h[i]+y1[i]
        prediction_box[upper_src_index[i]].append(src_box)
        prediction_score[upper_src_index[i]].append(src_score)
        prediction_label[upper_src_index[i]].append(j)
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
            
            label =person_object[prediction_label[i][j]] + ' score : '+str(int(prediction_score[i][j][k]*100)/100)
            csv_data['x1'].append(int(prediction_box[i][j][k][0]))
            csv_data['y1'].append(int(prediction_box[i][j][k][1]))
            csv_data['x2'].append(int(prediction_box[i][j][k][2]))
            csv_data['y2'].append(int(prediction_box[i][j][k][3]))
            csv_data['class_name'].append(person_object[prediction_label[i][j]])
            csv_data['score'].append(prediction_score[i][j][k])
            if person_object[prediction_label[i][j]]=='upper':
                continue
            if person_object[prediction_label[i][j]]=='lower':
                continue
            labelSize = cv2.getTextSize(label, fontFace, fontScale, thickness)
            img=cv2.rectangle(img,pt1,pt2,(0,255,0),4)
            cv2.rectangle(img, pt1, (pt1[0]+labelSize[0][0],pt1[1]-labelSize[0][1]), (0,255,0), cv2.FILLED)
            cv2.putText(img, label, pt1, fontFace, fontScale, (0,0,0), thickness)
    all_data = pd.DataFrame(csv_data)
    all_data.to_csv(dst_folder_path+'csv/'+dst_list[i][0:dst_list[i].find('.')]+'.csv',index=False,encoding='cp949',header=False)
    cv2.imwrite(dst_folder_path+dst_list[i],img)

