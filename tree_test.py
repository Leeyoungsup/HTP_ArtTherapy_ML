
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
model_path='./model/'

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

model=load_model(model_path+'T_whole_model.h5')


file_list=os.listdir(folder_path)
src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
tree_object=np.array(['tree','crown','pillar','fruit','branch','leaf','root','annual_zone'])
dst_list=[]
dst_box=[]
dst_label=[]
dst_score=[]
for file_name in file_list:
    try:
        input1=cv2.imread(folder_path+file_name)
        input1=cv2.resize(input,(512,512))
        input=cv2.zeros((1024,1024,3))
        input[256:768,256:768]=input1
        input=np.reshape(input,(1,1024,1024,3))
        src_img=np.concatenate((src_img,input),axis=0)
        dst_list.append(file_name)
    except:
        print(file_name+'은 이미지 파일이 아닙니다')

boxes, scores, labels=model.predict(src_img)
detetion_index=np.where(scores!=-1)[0]
prediction_box,prediction_label,prediction_score=[],[],[]
pillar_src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
pillar_src_index=[]
pillar_ratio_w=[]
pillar_ratio_h=[]
x1,x2,y1,y2=[],[],[],[]
for i in range(len(dst_list)):
    class_label=np.unique(labels[i])
    class_label=np.delete(class_label,np.where(class_label==-1))
    dst_box=[]
    dst_label=[]
    dst_score=[]
    for j in class_label:
        class_index=np.where(labels[i]==j)[0]
        if j==1:
            src_box,src_score=nms(boxes[i][class_index], scores[i][class_index],1,0.5)
        elif j==2:
            src_box,src_score=nms(boxes[i][class_index], scores[i][class_index],1,0.5)
        else:
            src_box,src_score=nms(boxes[i][class_index], scores[i][class_index],99,0.5)
        dst_box.append(src_box)
        dst_score.append(src_score)
        dst_label.append(j)
    prediction_box.append(dst_box)
    prediction_score.append(dst_score)
    prediction_label.append(dst_label)

for i in range(len(dst_list)):
    pillar_index=prediction_label[i].index(2)
    if len(prediction_box[i][pillar_index])==0: 
        continue
    pillar_box=prediction_box[i][pillar_index][0]
    x1.append(pillar_box[0])
    y1.append(pillar_box[1])
    x2.append(pillar_box[2])
    y2.append(pillar_box[3])
    
    pillar_ratio_w.append((pillar_box[2]-pillar_box[0])/1024)
    pillar_ratio_h.append((pillar_box[3]-pillar_box[1])/1024)
    pillar_img = src_img[i][int(pillar_box[1]): int(pillar_box[3]), int(pillar_box[0]): int(pillar_box[2])]
    pillar_img=cv2.resize(pillar_img,(1024,1024))
    pillar_img=np.reshape(pillar_img,(1,1024,1024,3))
    pillar_src_img=np.concatenate((pillar_src_img,pillar_img),axis=0)
    pillar_src_index.append(i)

model=load_model(model_path+'T_pillar_model.h5')
pillar_boxes, pillar_scores, pillar_labels=model.predict(pillar_src_img)  

for i in range(len(pillar_src_index)):
    dst_box=[]
    dst_label=[]
    dst_score=[]
    class_index=np.where(pillar_labels[i]==7)[0]
    if len(class_index)==0:
        continue
    else:
        src_box,src_score=nms(pillar_boxes[i][class_index], pillar_scores[i][class_index],99,0.99)
        src_box[:,0]=(src_box[:,0])*pillar_ratio_w[i]+x1[i]
        src_box[:,1]=(src_box[:,1])*pillar_ratio_h[i]+y1[i]
        src_box[:,2]=(src_box[:,2])*pillar_ratio_w[i]+x1[i]
        src_box[:,3]=(src_box[:,3])*pillar_ratio_h[i]+y1[i]
        prediction_box[pillar_src_index[i]].append(src_box)
        prediction_score[pillar_src_index[i]].append(src_score)
        prediction_label[pillar_src_index[i]].append(7)
        
root_ratio_w=[]
root_ratio_h=[]
x1,x2,y1,y2=[],[],[],[]
root_src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
root_src_index=[]
for i in range(len(dst_list)):
    root_index=prediction_label[i].index(2)
    if len(prediction_box[i][root_index])==0: 
        continue
    
    root_box=prediction_box[i][root_index][0]
    if root_box[3]>990:
        continue
    x1.append(0)
    y1.append(int(3*(root_box[3]/4)))
    x2.append(1024)
    y2.append(1024)
    
    root_ratio_w.append(1)
    root_ratio_h.append((1024-int(3*(root_box[3]/4)))/1024)
    root_img = src_img[i][int(3*(root_box[3]/4)):,:]
    root_img=cv2.resize(root_img,(1024,1024))
    root_img=np.reshape(root_img,(1,1024,1024,3))
    root_src_img=np.concatenate((root_src_img,root_img),axis=0)
    root_src_index.append(i)

model=load_model(model_path+'T_root_model.h5')
root_boxes, root_scores, root_labels=model.predict(root_src_img)  

for i in range(len(root_src_index)):
    dst_box=[]
    dst_label=[]
    dst_score=[]
    class_index=np.where(root_labels[i]==6)[0]
    if len(class_index)==0:
        continue
    else:
        src_box,src_score=nms(root_boxes[i][class_index], root_scores[i][class_index],1,0.99)
        src_box[:,0]=(src_box[:,0])*root_ratio_w[i]+x1[i]
        src_box[:,1]=(src_box[:,1])*root_ratio_h[i]+y1[i]
        src_box[:,2]=(src_box[:,2])*root_ratio_w[i]+x1[i]
        src_box[:,3]=(src_box[:,3])*root_ratio_h[i]+y1[i]
        prediction_box[root_src_index[i]].append(src_box)
        prediction_score[root_src_index[i]].append(src_score)
        prediction_label[root_src_index[i]].append(6)
        
branch_ratio_w=[]
branch_ratio_h=[]
x1,x2,y1,y2=[],[],[],[]
branch_src_img=np.empty((0,1024,1024,3),dtype=np.uint8)
branch_src_index=[]
for i in range(len(dst_list)):
    branch_index=prediction_label[i].index(2)
    if len(prediction_box[i][branch_index])==0: 
        continue
    branch_box=prediction_box[i][branch_index][0]

    x1.append(0)
    y1.append(0)
    x2.append(1024)
    y2.append(int(branch_box[3]))
    
    branch_ratio_w.append(1)
    branch_ratio_h.append(int(branch_box[3])/1024)
    branch_img = src_img[i][:int(branch_box[3]),:]
    branch_img=cv2.resize(branch_img,(1024,1024))
    branch_img=np.reshape(branch_img,(1,1024,1024,3))
    branch_src_img=np.concatenate((branch_src_img,branch_img),axis=0)
    branch_src_index.append(i)

model=load_model(model_path+'T_branch_model.h5')
branch_boxes, branch_scores, branch_labels=model.predict(branch_src_img)  

for i in range(len(branch_src_index)):
    dst_box=[]
    dst_label=[]
    dst_score=[]
    class_index=np.where(branch_labels[i]==4)[0]
    if len(class_index)==0:
        continue
    else:
        src_box,src_score=nms(branch_boxes[i][class_index], branch_scores[i][class_index],20,0.99)
        src_box[:,0]=(src_box[:,0])*branch_ratio_w[i]+x1[i]
        src_box[:,1]=(src_box[:,1])*branch_ratio_h[i]+y1[i]
        src_box[:,2]=(src_box[:,2])*branch_ratio_w[i]+x1[i]
        src_box[:,3]=(src_box[:,3])*branch_ratio_h[i]+y1[i]
        prediction_box[branch_src_index[i]].append(src_box)
        prediction_score[branch_src_index[i]].append(src_score)
        prediction_label[branch_src_index[i]].append(4)

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
            if tree_object[prediction_label[i][j]]=='crown' and int(prediction_box[i][j][k][3])>7*1024/8:
                continue
            else:
                img=cv2.rectangle(img,pt1,pt2,(0,255,0),4)
                label =tree_object[prediction_label[i][j]] + ' score : '+str(int(prediction_score[i][j][k]*100)/100)
                csv_data['x1'].append(int(prediction_box[i][j][k][0]))
                csv_data['y1'].append(int(prediction_box[i][j][k][1]))
                csv_data['x2'].append(int(prediction_box[i][j][k][2]))
                csv_data['y2'].append(int(prediction_box[i][j][k][3]))
                csv_data['class_name'].append(tree_object[prediction_label[i][j]])
                csv_data['score'].append(prediction_score[i][j][k])
                labelSize = cv2.getTextSize(label, fontFace, fontScale, thickness)
                cv2.rectangle(img, pt1, (pt1[0]+labelSize[0][0],pt1[1]-labelSize[0][1]), (0,255,0), cv2.FILLED)
                cv2.putText(img, label, pt1, fontFace, fontScale, (0,0,0), thickness)
    all_data = pd.DataFrame(csv_data)
    all_data.to_csv(dst_folder_path+'csv/'+dst_list[i][0:dst_list[i].find('.')]+'.csv',index=False,encoding='cp949',header=False)
    cv2.imwrite(dst_folder_path+dst_list[i],img)
