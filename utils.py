'''一些工具'''
from PIL import Image
import numpy
import os
import time
import random
import math
from matplotlib import pyplot

def segment(from_path,to_path,thre):
    '''将模型的预测图进行阈值分割
    阈值要小数，而且是 thre*(max-min)+min
    '''
    d=os.listdir(from_path)
    for i in range(len(d)):
        predict=numpy.array(Image.open(from_path+'\\'+d[i]))
        ma=predict.max()
        mi=predict.min()
        t=thre*(ma-mi)+mi
        img=numpy.where(predict>t,255,0).astype(numpy.uint8)
        Image.fromarray(img).save(to_path+'\\'+d[i])
        print(i)
        
def merge_segement(from_paths,to_path,thres):
    '''from_paths 一个列表 [预测图的文件夹，......]
    to_path 存合并后的文件夹
    将多个模型生成的预测图,阈值分割后，合并成一张图。
    赋予每个像素 标签值相加/模型数
    比如，有三个模型，某像素有一个模型认为时水（1），另外两个认为是非水体（0），则赋予  （0+0+1）/3
    '''
    result=numpy.zeros((256,256))
    d=os.listdir(from_paths[0])
    for i in range(len(d)):
        sta=numpy.zeros((256,256))
        for j in range(len(from_paths)):
            predict=numpy.array(Image.open(from_paths[j]+'\\'+d[i]))
            ma=predict.max()
            mi=predict.min()
            t=thres[j]*(ma-mi)+mi
            img=numpy.where(predict>t,1,0)
            sta=sta+img
        sta=sta/len(from_paths)
        Image.fromarray((sta*255).astype(numpy.uint8)).save(to_path+'\\'+d[i])
        print(i)
        
def window_smooth(from_path,to_path,window_size):
    '''对于不确定的像素，取一定的窗口，窗口内平均的标签值，作为该像素标签值
    输入是merge函数的输出
    '''
    d=os.listdir(from_path)
    s=int(window_size/2)
    for k in range(len(d)):
        if k%100==0:
            print(k)
        img=numpy.array(Image.open(from_path+'\\'+d[k]))
        result=numpy.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                left=j-s
                if left<0:
                    left=0
                right=j+s
                if right>=img.shape[1]:
                    right=img.shape[1]-1
                top=i-s
                if top<0:
                    top=0
                bottom=i+s
                if bottom>=img.shape[0]:
                    bottom=img.shape[0]-1
                
                result[i,j]=img[top:bottom+1,left:right+1].mean()
                
        Image.fromarray(result).save(to_path+'\\'+d[k])

        
def random_label(from_path,to_path):
    '''对于merge 出来的标签，将其标签值作为水体的概率进行随机（非0或1时）
    比如某像素merge后的值为0.25，则随机赋予其0或1的标签，赋予0概率为0.75  1概率为0.25
    保存0-255
    
    '''
    d=os.listdir(from_path)
    for i in range(len(d)):
        if i%100==0:
            print(i)
        img=numpy.array(Image.open(from_path+'\\'+d[i]))/255
        for j in range(256):
            for k in range(256):
                if img[j,k]!=0 and img[j,k]!=1:
                    flag=random.random()
                    if flag<img[j,k]:
                        img[j,k]=1
                    else:
                        img[j,k]=0
        Image.fromarray((img*255).astype(numpy.uint8)).save(to_path+'\\'+d[i])
                    
                        
def mask(from_path,to_path):
    '''生成mask，对于不确定的像素，赋予0，确定的赋予1
    从merge中获得
    保存0-1
    '''
    d=os.listdir(from_path)
    for i in range(len(d)):
        if i%100==0:
            print(i)
        img=numpy.array(Image.open(from_path+'\\'+d[i]))/255
        

        
        mask1=numpy.where(img==1,True,False)
        mask2=numpy.where(img==0,True,False)
        mask=mask1+mask2
        mask=numpy.where(mask,1,0)
        
        Image.fromarray(mask).save(to_path+'\\'+d[i])
    
def changename(path,name):
    '''将一个文件夹内文件名 中间的 替换成 name  （比如 predict lable lc s1.。。）
    '''
    d=os.listdir(path)
    for i in range(len(d)):
        if i%100==0:
            print(i)
        new=d[i].split('_')
        new[2]=name
        new='_'.join(new)
        os.rename(path+'\\'+d[i],path+'\\'+new)
        
def caculate_square_error(path_predict,path_label,path_mask=''):
    '''计算预测和标签的差的平方和的平均值开根
    mask 是 0-1   是1则参与计算
    '''
    d1=os.listdir(path_predict)
    d2=os.listdir(path_label)
    if path_mask:
        d3=os.listdir(path_mask)
        name_mid3=d3[0].split('_')[2]
    result=numpy.zeros(len(d1))
    name_mid2=d2[0].split('_')[2]
    for i in range(len(d1)):
        img1=numpy.array(Image.open(path_predict+'\\'+d1[i]))/255
        name2=d1[i].split('_')
        name2[2]=name_mid2
        name2='_'.join(name2)
        img2=numpy.array(Image.open(path_label+'\\'+name2))/255
        
        if path_mask:
            name3=d1[i].split('_')
            name3[2]=name_mid3
            name3='_'.join(name3)
            mask=numpy.array(Image.open(path_mask+'\\'+name3))
        else:
            mask=numpy.ones((256,256))
        t=(img1-img2)*(img1-img2)*mask
        result[i]=t.sum()
        result[i]=result[i]/(mask.sum())
        result[i]=math.sqrt(result[i])
    
    return result.sum()/result.shape[0]
        
def masked_label(path_predict,path_mask,path_to):
    '''将mask中为0的像素的标签设为0，为1的保留原有标签
    path_predict  原有标签
    path_mask mask
    '''
    d1=os.listdir(path_predict)
    d2=os.listdir(path_mask)
    name_mid2=d2[0].split('_')[2]
    for i in range(len(d1)):
        img1=numpy.array(Image.open(path_predict+'\\'+d1[i]))/255
        name2=d1[i].split('_')
        name2[2]=name_mid2
        name2='_'.join(name2)
        img2=numpy.array(Image.open(path_mask+'\\'+name2))

        result=img1*img2
        Image.fromarray(result).save(path_to+'\\'+d1[i])
        
def random_plus_merge(path_merge,path_random,path_to):
    '''将merge 与 random 相乘 希望结合两者的优点
    '''
    d1=os.listdir(path_merge)
    d2=os.listdir(path_random)
    name_mid2=d2[0].split('_')[2]
    for i in range(len(d1)):
        img1=numpy.array(Image.open(path_merge+'\\'+d1[i]))/255
        name2=d1[i].split('_')
        name2[2]=name_mid2
        name2='_'.join(name2)
        img2=numpy.array(Image.open(path_random+'\\'+name2))/255

        result=(img1*img2*255).astype(numpy.uint8)
        Image.fromarray(result).save(path_to+'\\'+d1[i])
        
'''
记录地理坐标，可以展示数据分布
测试时可以取平均
尝试多种波段
评价指标多种
淘宝整个gpu
非水体全加上
'''
        
if __name__=='__main__':
    path_predict_model1='D:\\Codes\\test_dataset\\1ite\\predict_1_model1'
    path_predict_model2='D:\\Codes\\test_dataset\\1ite\\predict_1_model2'
    path_predict_model3='D:\\Codes\\test_dataset\\1ite\\predict_1_model3'
    path_merge='D:\\Codes\\test_dataset\\1ite\\predict_1_merge'
    path_random='D:\\Codes\\test_dataset\\1ite\\predict_1_random'
    path_label='D:\\Codes\\test_dataset\\label'
    path_mask='D:\\Codes\\test_dataset\\1ite\\predict_1_mask'
    path_merge_smooth='D:\\Codes\\test_dataset\\1ite\\predict_1_merge_smooth_5'
    path_merge_mask='D:\\Codes\\test_dataset\\1ite\\predict_1_merge_mask'
    path_random_merge='D:\\Codes\\test_dataset\\1ite\\predict_1_random_merge'
    thres=[0.31,0.24,0.34]
    window_size=5
    
    #random_plus_merge(path_merge,path_random,path_random_merge)
    #masked_label(path_merge,path_mask,path_merge_mask)
    segment('G:\\Graguation\\test_result\\1ite\\pre','G:\\Graguation\\test_result\\1ite\\pre_seg',0.3)
    #err=caculate_square_error(path_random_merge,path_label,path_mask)
    #print(err)
    #window_smooth(path_merge,path_merge_smooth,window_size)
    #mask(path_merge,path_mask)
    #merge_segement([path_predict_model1,path_predict_model2,path_predict_model3],path_merge,thres)
    #random_label(path_merge,path_random)
    '''
    err1=caculate_square_error(path_random,path_label,path_mask)
    err2=caculate_square_error(path_random,path_label)
    err3=caculate_square_error(path_merge,path_label)
    pyplot.plot(err1,label='mask+random')
    pyplot.plot(err2,label='random')
    pyplot.plot(err3,label='merge')
    pyplot.legend()
    pyplot.show()
    '''
