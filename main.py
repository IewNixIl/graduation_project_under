from Config import config
from train import train
from test import testTools,getimg,getlabel_low
import numpy
from DATA import WaterDataset
import Networks
from torchvision import transforms as T
from torch.utils.data import DataLoader
from matplotlib import pyplot
import os
from PIL import Image
import time
import util
from utils import correct_merge
from multiprocessing import Process
import torch


def predict_seg_merge_mask(models,n_ite,thres,ifrun=[True,True,True]):
    path_models='D:\\codes\\Graduation\\MODELS'
    subs=['train_sub1','train_sub2','train_sub3']#subs 和 models 对应
    path='D:\\result\\train_result'
    transform_img=config.transform_img
    transform_label=config.transform_label




    model1=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    path_model=path_models+'\\'+models[0]
    #加载模型
    model1.load(path_model)
    if config.use_gpu:#使用gpu
        model1=model1.cuda()

    model2=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    path_model=path_models+'\\'+models[1]
    #加载模型
    model2.load(path_model)
    if config.use_gpu:#使用gpu
        model2=model2.cuda()

    model3=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    path_model=path_models+'\\'+models[2]
    #加载模型
    model3.load(path_model)
    if config.use_gpu:#使用gpu
        model3=model3.cuda()

    test_data=WaterDataset(sub=subs[0],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader1=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)

    test_data=WaterDataset(sub=subs[1],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader2=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)

    test_data=WaterDataset(sub=subs[2],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader3=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)

    if ifrun[0]:
        name_merge='predict_'+str(n_ite)+'_sub1_merge'
        name_mask='predict_'+str(n_ite)+'_sub1_mask'
        path_merge=path+'\\'+str(n_ite)+'ite'+'\\'+name_merge
        path_mask=path+'\\'+str(n_ite)+'ite'+'\\'+name_mask
        if not os.path.exists(path_merge):
            os.makedirs(path_merge)
        if not os.path.exists(path_mask):
            os.makedirs(path_mask)
        for i,data in enumerate(test_dataloader1,0):

            inputs,labels,name=data#获得输入和标签
            if config.use_gpu:#使用gpu
                inputs=inputs.cuda()


            merge=(torch.where(model1(inputs)>torch.tensor(thres[0]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda())+
            torch.where(model2(inputs)>torch.tensor(thres[1]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda())+
            torch.where(model3(inputs)>torch.tensor(thres[2]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda()))/torch.tensor([3]).cuda()

            merge=numpy.array(merge[0,0,:,:].cpu().detach())

            mask=numpy.where(numpy.where(merge==1,True,False)+numpy.where(merge==0,True,False),1,0)

            Image.fromarray((merge*255).astype(numpy.uint8)).save(path+'\\'+str(n_ite)+'ite'+'\\'+name_merge+'\\'+name[0])
            Image.fromarray(mask).save(path+'\\'+str(n_ite)+'ite'+'\\'+name_mask+'\\'+name[0])

            if i%1000==0:
                print(i)

    if ifrun[1]:
        name_merge='predict_'+str(n_ite)+'_sub2_merge'
        name_mask='predict_'+str(n_ite)+'_sub2_mask'
        path_merge=path+'\\'+str(n_ite)+'ite'+'\\'+name_merge
        path_mask=path+'\\'+str(n_ite)+'ite'+'\\'+name_mask
        if not os.path.exists(path_merge):
            os.makedirs(path_merge)
        if not os.path.exists(path_mask):
            os.makedirs(path_mask)
        for i,data in enumerate(test_dataloader2,0):

            inputs,labels,name=data#获得输入和标签
            if config.use_gpu:#使用gpu
                inputs=inputs.cuda()


            merge=(torch.where(model1(inputs)>torch.tensor(thres[0]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda())+
            torch.where(model2(inputs)>torch.tensor(thres[1]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda())+
            torch.where(model3(inputs)>torch.tensor(thres[2]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda()))/torch.tensor([3]).cuda()

            merge=numpy.array(merge[0,0,:,:].cpu().detach())

            mask=numpy.where(numpy.where(merge==1,True,False)+numpy.where(merge==0,True,False),1,0)

            Image.fromarray((merge*255).astype(numpy.uint8)).save(path+'\\'+str(n_ite)+'ite'+'\\'+name_merge+'\\'+name[0])
            Image.fromarray(mask).save(path+'\\'+str(n_ite)+'ite'+'\\'+name_mask+'\\'+name[0])

            if i%1000==0:
                print(i)

    if ifrun[2]:
        name_merge='predict_'+str(n_ite)+'_sub3_merge'
        name_mask='predict_'+str(n_ite)+'_sub3_mask'
        path_merge=path+'\\'+str(n_ite)+'ite'+'\\'+name_merge
        path_mask=path+'\\'+str(n_ite)+'ite'+'\\'+name_mask
        if not os.path.exists(path_merge):
            os.makedirs(path_merge)
        if not os.path.exists(path_mask):
            os.makedirs(path_mask)
        for i,data in enumerate(test_dataloader3,0):

            inputs,labels,name=data#获得输入和标签
            if config.use_gpu:#使用gpu
                inputs=inputs.cuda()


            merge=(torch.where(model1(inputs)>torch.tensor(thres[0]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda())+
            torch.where(model2(inputs)>torch.tensor(thres[1]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda())+
            torch.where(model3(inputs)>torch.tensor(thres[2]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda()))/torch.tensor([3]).cuda()

            merge=numpy.array(merge[0,0,:,:].cpu().detach())

            mask=numpy.where(numpy.where(merge==1,True,False)+numpy.where(merge==0,True,False),1,0)

            Image.fromarray((merge*255).astype(numpy.uint8)).save(path+'\\'+str(n_ite)+'ite'+'\\'+name_merge+'\\'+name[0])
            Image.fromarray(mask).save(path+'\\'+str(n_ite)+'ite'+'\\'+name_mask+'\\'+name[0])

            if i%1000==0:
                print(i)




def predict_all(models,n_ite,ifrun):
    '''已有三个sub数据
    三个训练好的模型
    生成九个预测 （相互交叉）
    '''
    path_models='D:\\codes\\Graduation\\MODELS'
    #models=['model35','model36','model37']
    subs=['train_sub1','train_sub2','train_sub3']#subs 和 models 对应
    #n_ite=1#从1开始计数
    path_pre='D:\\result\\train_result'


    transform_img=config.transform_img
    transform_label=config.transform_label


    #模型设置
    model=getattr(Networks,config.model)(config.input_band,1)#创立网络对象


    if ifrun[0]:
        '''model1 预测sub1 sub2 sub3'''
        path_model=path_models+'\\'+models[0]
        #加载模型
        model.load(path_model)
        if config.use_gpu:#使用gpu
            model=model.cuda()

        #sub1
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub1_model1'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[0],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)

        #sub2
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub2_model1'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[1],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)

        #sub3
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub3_model1'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[2],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)


    '''model2 预测sub1 sub2 sub3'''
    if ifrun[1]:
        path_model=path_models+'\\'+models[1]
        #加载模型
        model.load(path_model)
        if config.use_gpu:#使用gpu
            model=model.cuda()

        #sub1
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub1_model2'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[0],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)

        #sub2
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub2_model2'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[1],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)

        #sub3
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub3_model2'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[2],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)

    '''model3 预测sub1 sub2 sub3'''
    if ifrun[2]:
        path_model=path_models+'\\'+models[2]
        #加载模型
        model.load(path_model)
        if config.use_gpu:#使用gpu
            model=model.cuda()

        #sub1
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub1_model3'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[0],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)

        #sub2
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub2_model3'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[1],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)

        #sub3
        path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub3_model3'
        if not os.path.exists(path_predict):
            os.makedirs(path_predict)
        test_data=WaterDataset(sub=subs[2],train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)
        test_dataloader=DataLoader(test_data,1,
                                shuffle=True,#每一epoch打乱数据
                                num_workers=config.num_workers)
        t=testTools(model=model,data=test_dataloader)
        t.predict(path_predict=path_predict)

def segment_merge(n_ite,thre,ifrun):
    '''分割+合并
    分割  用一定的阈值
    合并 赋予每个像素 标签值相加/模型数  比如，有三个模型，某像素有一个模型认为时水（1），另外两个认为是非水体（0），则赋予  （0+0+1）/3
    '''

    #n_ite=1
    path='D:\\result\\train_result\\'+str(n_ite)+'ite'
    #thre=[0.31,0.24,0.34]#对应 model1 2 3

    if ifrun[0]:
        '''sub1'''
        name=['predict_'+str(n_ite)+'_sub1_model1' , 'predict_'+str(n_ite)+'_sub1_model2' , 'predict_'+str(n_ite)+'_sub1_model3']
        name_to='predict_'+str(n_ite)+'_sub1_merge'
        if not os.path.exists(path+'\\'+name_to):
            os.makedirs(path+'\\'+name_to)
        d=os.listdir(path+'\\'+name[0])
        for i in range(len(d)):
            if i%100==0:
                print(i)

            predict1=numpy.array(Image.open(path+'\\'+name[0]+'\\'+d[i]))
            predict2=numpy.array(Image.open(path+'\\'+name[1]+'\\'+d[i]))
            predict3=numpy.array(Image.open(path+'\\'+name[2]+'\\'+d[i]))


            img1=numpy.where(predict1>thre[0],1,0)

            img2=numpy.where(predict2>thre[1],1,0)


            img3=numpy.where(predict3>thre[2],1,0)

            img=(img1+img2+img3)/3
            Image.fromarray((img*255).astype(numpy.uint8)).save(path+'\\'+name_to+'\\'+d[i])

    if ifrun[1]:
        '''sub2'''
        name=['predict_'+str(n_ite)+'_sub2_model1' , 'predict_'+str(n_ite)+'_sub2_model2' , 'predict_'+str(n_ite)+'_sub2_model3']
        name_to='predict_'+str(n_ite)+'_sub2_merge'
        if not os.path.exists(path+'\\'+name_to):
            os.makedirs(path+'\\'+name_to)
        d=os.listdir(path+'\\'+name[0])
        for i in range(len(d)):
            if i%100==0:
                print(i)
            predict1=numpy.array(Image.open(path+'\\'+name[0]+'\\'+d[i]))
            predict2=numpy.array(Image.open(path+'\\'+name[1]+'\\'+d[i]))
            predict3=numpy.array(Image.open(path+'\\'+name[2]+'\\'+d[i]))

            img1=numpy.where(predict1>thre[0],1,0)

            img2=numpy.where(predict2>thre[1],1,0)


            img3=numpy.where(predict3>thre[2],1,0)

            img=(img1+img2+img3)/3
            Image.fromarray((img*255).astype(numpy.uint8)).save(path+'\\'+name_to+'\\'+d[i])

    if ifrun[2]:
        '''sub3'''
        name=['predict_'+str(n_ite)+'_sub3_model1' , 'predict_'+str(n_ite)+'_sub3_model2' , 'predict_'+str(n_ite)+'_sub3_model3']
        name_to='predict_'+str(n_ite)+'_sub3_merge'
        if not os.path.exists(path+'\\'+name_to):
            os.makedirs(path+'\\'+name_to)
        d=os.listdir(path+'\\'+name[0])
        for i in range(len(d)):
            if i%100==0:
                print(i)
            predict1=numpy.array(Image.open(path+'\\'+name[0]+'\\'+d[i]))
            predict2=numpy.array(Image.open(path+'\\'+name[1]+'\\'+d[i]))
            predict3=numpy.array(Image.open(path+'\\'+name[2]+'\\'+d[i]))

            img1=numpy.where(predict1>thre[0],1,0)

            img2=numpy.where(predict2>thre[1],1,0)


            img3=numpy.where(predict3>thre[2],1,0)

            img=(img1+img2+img3)/3
            Image.fromarray((img*255).astype(numpy.uint8)).save(path+'\\'+name_to+'\\'+d[i])


def random_labels():
    '''对utils里的random_label 的集成
    '''
    n_ite=1
    path='G:\\Graguation\\train_result\\'+str(n_ite)+'ite'
    name_from=['predict_'+str(n_ite)+'_sub1_merge','predict_'+str(n_ite)+'_sub2_merge','predict_'+str(n_ite)+'_sub3_merge']
    name_to=['predict_'+str(n_ite)+'_sub1_random','predict_'+str(n_ite)+'_sub2_random','predict_'+str(n_ite)+'_sub3_random']

    '''sub1'''
    util.random_label(path+'\\'+name_from[0],path+'\\'+name_to[0])

    '''sub2'''
    util.random_label(path+'\\'+name_from[1],path+'\\'+name_to[1])

    '''sub3'''
    util.random_label(path+'\\'+name_from[2],path+'\\'+name_to[2])

def masks(n_ite,ifrun):
    '''对utils里的mask 的集成
    '''
    #n_ite=1
    path='D:\\result\\train_result\\'+str(n_ite)+'ite'
    name_from=['predict_'+str(n_ite)+'_sub1_merge','predict_'+str(n_ite)+'_sub2_merge','predict_'+str(n_ite)+'_sub3_merge']
    name_to=['predict_'+str(n_ite)+'_sub1_mask_correct','predict_'+str(n_ite)+'_sub2_mask_correct','predict_'+str(n_ite)+'_sub3_mask_correct']

    if ifrun[0]:
        '''sub1'''
        if not os.path.exists(path+'\\'+name_to[0]):
            os.makedirs(path+'\\'+name_to[0])
        util.mask(path+'\\'+name_from[0],path+'\\'+name_to[0])

    if ifrun[1]:
        '''sub2'''
        if not os.path.exists(path+'\\'+name_to[1]):
            os.makedirs(path+'\\'+name_to[1])
        util.mask(path+'\\'+name_from[1],path+'\\'+name_to[1])

    if ifrun[2]:
        '''sub3'''
        if not os.path.exists(path+'\\'+name_to[2]):
            os.makedirs(path+'\\'+name_to[2])
        util.mask(path+'\\'+name_from[2],path+'\\'+name_to[2])

def masks_another(n_ite,ifrun):
    '''对utils里的mask_another 的集成
    '''
    #n_ite=1
    path='D:\\result\\train_result\\'+str(n_ite)+'ite'
    name_from=['predict_'+str(n_ite)+'_sub1_merge','predict_'+str(n_ite)+'_sub2_merge','predict_'+str(n_ite)+'_sub3_merge']
    name_to=['predict_'+str(n_ite)+'_sub1_mask_another','predict_'+str(n_ite)+'_sub2_mask_another','predict_'+str(n_ite)+'_sub3_mask_another']

    if ifrun[0]:
        '''sub1'''
        if not os.path.exists(path+'\\'+name_to[0]):
            os.makedirs(path+'\\'+name_to[0])
        util.mask_another(path+'\\'+name_from[0],path+'\\'+name_to[0])

    if ifrun[1]:
        '''sub2'''
        if not os.path.exists(path+'\\'+name_to[1]):
            os.makedirs(path+'\\'+name_to[1])
        util.mask_another(path+'\\'+name_from[1],path+'\\'+name_to[1])

    if ifrun[2]:
        '''sub3'''
        if not os.path.exists(path+'\\'+name_to[2]):
            os.makedirs(path+'\\'+name_to[2])
        util.mask_another(path+'\\'+name_from[2],path+'\\'+name_to[2])

def merge_mask():
    '''utils 里的 masked_label 的集成
    '''
    n_ite=1
    path='G:\\Graguation\\train_result\\'+str(n_ite)+'ite'
    name_from=['predict_'+str(n_ite)+'_sub1_merge','predict_'+str(n_ite)+'_sub2_merge','predict_'+str(n_ite)+'_sub3_merge']
    name_mask=['predict_'+str(n_ite)+'_sub1_mask','predict_'+str(n_ite)+'_sub2_mask','predict_'+str(n_ite)+'_sub3_mask']
    name_to=['predict_'+str(n_ite)+'_sub1_merge_mask','predict_'+str(n_ite)+'_sub2_merge_mask','predict_'+str(n_ite)+'_sub3_merge_mask']

    '''sub1'''
    util.masked_label(path+'\\'+name_from[0],path+'\\'+name_mask[0],path+'\\'+name_to[0])

    '''sub2'''
    util.masked_label(path+'\\'+name_from[1],path+'\\'+name_mask[1],path+'\\'+name_to[1])

    '''sub3'''
    util.masked_label(path+'\\'+name_from[2],path+'\\'+name_mask[2],path+'\\'+name_to[2])


def correct_all(n_ite,ifrun):
    '''对utils里的correct_merge 的集成
    '''
    #n_ite=1
    path='D:\\result\\train_result\\'+str(n_ite)+'ite'
    name_from=['predict_'+str(n_ite)+'_sub1_merge','predict_'+str(n_ite)+'_sub2_merge','predict_'+str(n_ite)+'_sub3_merge']
    name_to=['predict_'+str(n_ite)+'_sub1_correct','predict_'+str(n_ite)+'_sub2_correct','predict_'+str(n_ite)+'_sub3_correct']

    if ifrun[0]:
        '''sub1'''
        if not os.path.exists(path+'\\'+name_to[0]):
            os.makedirs(path+'\\'+name_to[0])
        correct_merge(path+'\\'+name_from[0],path+'\\'+name_to[0])

    if ifrun[1]:
        '''sub2'''
        if not os.path.exists(path+'\\'+name_to[1]):
            os.makedirs(path+'\\'+name_to[1])
        correct_merge(path+'\\'+name_from[1],path+'\\'+name_to[1])

    if ifrun[2]:
        '''sub3'''
        if not os.path.exists(path+'\\'+name_to[2]):
            os.makedirs(path+'\\'+name_to[2])
        correct_merge(path+'\\'+name_from[2],path+'\\'+name_to[2])


def train_test(path_model,path_sta,path_label,path_mask,sub_name,ifboard=config.ifboard):
    train(path_model,path_sta,path_label=path_label,path_mask=path_mask,sub_name=sub_name,ifboard=ifboard)
    '''模型设置'''
    model=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    #加载模型
    model.load(path_model)
    if config.use_gpu:#使用gpu
        model=model.cuda()

    '''数据加载'''
    transform_img=config.transform_img
    transform_label=config.transform_label
    test_data=WaterDataset(train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)


    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=2)



    t=testTools(model=model,data=test_dataloader,path_model=path_model)
    #t.predict()
    #t.pr(save=save,show=show)
    #t.predict(0.24)


    #t.getlabel_low_trained(0.48)
    #getlabel_low()


    p,r,f=t.pr_fmeasure()
    pa=t.PA()
    iou=t.IoU()
    miou=t.MIoU()
    aa=t.AA()

    #t.drawStatistic()
    #p,r,f,pa,iou=t.integrate('F:\\Graduation\\MODELS',['model20','model21','model22','model23'])


    l=iou
    thre=int(l.argmax())
    print('阈值为'+str(thre/100)+'时')
    print('precision: '+str(p[thre]))
    print('recall: '+str(r[thre]))
    print('f measure: '+str(f[thre]))
    print('pa: '+str(pa[thre]))
    print('iou: '+str(iou[thre]))
    print('miou: '+str(miou[thre]))
    print('aa: '+str(aa[thre]))
    t.recordPrecision(thre/100,p[thre],r[thre],f[thre],pa[thre],iou[thre])

    '''
    pyplot.plot(f,label='f-measure')
    pyplot.plot(pa,label='pa')
    pyplot.plot(iou,label='iou')
    pyplot.legend()
    pyplot.show()
    '''

    #t.drawStatistic()

    return thre/100

def test3(models):
    '''模型设置'''

    model1=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    model2=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    model3=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    #加载模型
    model1.load('D:\\codes\\Graduation\\MODELS\\'+models[0])
    model2.load('D:\\codes\\Graduation\\MODELS\\'+models[1])
    model3.load('D:\\codes\\Graduation\\MODELS\\'+models[2])
    if config.use_gpu:#使用gpu
        model1=model1.cuda()
        model2=model2.cuda()
        model3=model3.cuda()

    '''数据加载'''

    transform_img=config.transform_img
    transform_label=config.transform_label
    test_data=WaterDataset(train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)


    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=2)

    #getsar('G:\\Graguation\\test_result\\sar1','G:\\Graguation\\test_result\\sar2')
    #getlabel_low('G:\\Graguation\\test_result\\lowlabel')
    t=testTools(model=[model1,model2,model3],data=test_dataloader)
    #t.predict()
    #t.pr(save=save,show=show)
    #t.predict(0.24)


    #t.getlabel_low_trained(0.48)
    #getlabel_low()
    #t.drawStatistic()

    p,r,f=t.pr_fmeasure()
    pa=t.PA()
    iou=t.IoU()
    miou=t.MIoU()
    aa=t.AA()

    #t.drawStatistic()
    #p,r,f,pa,iou=t.integrate('F:\\Graduation\\MODELS',['model20','model21','model22','model23'])


    l=iou
    thre=int(l.argmax())
    print('阈值为'+str(thre/100)+'时')
    print('precision: '+str(p[thre]))
    print('recall: '+str(r[thre]))
    print('f measure: '+str(f[thre]))
    print('oa: '+str(pa[thre]))
    print('iou: '+str(iou[thre]))
    print('miou: '+str(miou[thre]))
    print('aa: '+str(aa[thre]))

    #t.recordPrecision(thre/100,p[thre],r[thre],f[thre],pa[thre],iou[thre])




    pyplot.plot(f,label='f-measure')
    pyplot.plot(pa,label='pa')
    pyplot.plot(iou,label='iou')
    pyplot.legend()
    pyplot.show()


if __name__=='__main__':
    #getlabel_low('D:\\result\\final_result\\labels_low')
    test3(['model67','model68','model69'])
    #util.segment('D:\\result\\final_result\\predicts','D:\\result\\final_result\\segments',0.38)
    #getimg()


    #ifrun=[True,False,False]
    #ifrun=[False,True,False]
    #ifrun=[False,False,True]
    #ifrun=[True,True,True]
    #n_ite=2


    #predict_all(['model19','model20','model21'],n_ite,ifrun)
    #predict_all(['model4','model5','model6'],2,[False,True,False])
    #predict_all(['model4','model5','model6'],2,[False,False,True])
    #segment_merge(4,[0.38,0.37,0.24],[True,True,True])
    #masks_another(4,[True,True,True])
    #correct_all(1,ifrun)

    #predict_seg_merge_mask(['model43','model44','model45'],n_ite,[0.03,0.04,0.2])
    #segment_merge()
    #random_labels()
    #masks()
    #train_test()
    '''
    from multiprocessing import Process
    t1=time.time()
    p1 = Process(target=predict_all, args=(['model40','model41','model42'],n_ite,[False,False,True],))
    p2 = Process(target=predict_all, args=(['model40','model41','model42'],n_ite,[False,True,False],))
    p3 = Process(target=predict_all, args=(['model40','model41','model42'],n_ite,[True,False,False],))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    t2=time.time()
    print((t2-t1)/60)


    t1=time.time()
    p1 = Process(target=segment_merge, args=(n_ite,[0.15,0.07,0.06],[False,False,True],))
    p2 = Process(target=segment_merge, args=(n_ite,[0.15,0.07,0.06],[False,True,False],))
    p3 = Process(target=segment_merge, args=(n_ite,[0.15,0.07,0.06],[True,False,False],))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    t2=time.time()
    print((t2-t1)/60)


    t1=time.time()
    p1 = Process(target=correct_all, args=(n_ite,[False,False,True],))
    p2 = Process(target=correct_all, args=(n_ite,[False,True,False],))
    p3 = Process(target=correct_all, args=(n_ite,[True,False,False],))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    t2=time.time()
    print((t2-t1)/60)


    t1=time.time()
    p1 = Process(target=masks, args=(n_ite,[False,False,True],))
    p2 = Process(target=masks, args=(n_ite,[False,True,False],))
    p3 = Process(target=masks, args=(n_ite,[True,False,False],))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    t2=time.time()
    print((t2-t1)/60)
    '''
    #predict_seg_merge_mask(['model61','model62','model63'],3,[0.09,0.15,0.14],ifrun=[False,True,True])
    '''
    models=['67','68','69']
    n_ite_train=5

    path_model='D:\\codes\\Graduation\\MODELS\\model'+models[0]
    path_sta='D:\\codes\\Graduation\\MODELS\\sta'+models[0]
    path_label='D:\\result\\train_result\\'+str(n_ite_train-1)+'ite\\predict_'+str(n_ite_train-1)+'_sub1_merge'
    path_mask='D:\\result\\train_result\\'+str(n_ite_train-1)+'ite\\predict_'+str(n_ite_train-1)+'_sub1_mask'
    sub_name='train_sub1'
    util.changename(path_label,'label')
    util.changename(path_mask,'label')
    t1=train_test(path_model,path_sta,path_label,path_mask,sub_name)

    path_model='D:\\codes\\Graduation\\MODELS\\model'+models[1]
    path_sta='D:\\codes\\Graduation\\MODELS\\sta'+models[1]
    path_label='D:\\result\\train_result\\'+str(n_ite_train-1)+'ite\\predict_'+str(n_ite_train-1)+'_sub2_merge'
    path_mask='D:\\result\\train_result\\'+str(n_ite_train-1)+'ite\\predict_'+str(n_ite_train-1)+'_sub2_mask'
    sub_name='train_sub2'
    util.changename(path_label,'label')
    util.changename(path_mask,'label')
    t2=train_test(path_model,path_sta,path_label,path_mask,sub_name)

    path_model='D:\\codes\\Graduation\\MODELS\\model'+models[2]
    path_sta='D:\\codes\\Graduation\\MODELS\\sta'+models[2]
    path_label='D:\\result\\train_result\\'+str(n_ite_train-1)+'ite\\predict_'+str(n_ite_train-1)+'_sub3_merge'
    path_mask='D:\\result\\train_result\\'+str(n_ite_train-1)+'ite\\predict_'+str(n_ite_train-1)+'_sub3_mask'
    sub_name='train_sub3'
    util.changename(path_label,'label')
    util.changename(path_mask,'label')
    t3=train_test(path_model,path_sta,path_label,path_mask,sub_name)

    #predict_seg_merge_mask(['model'+models[0],'model'+models[1],'model'+models[2]],n_ite_train,[t1,t2,t3])
    '''
