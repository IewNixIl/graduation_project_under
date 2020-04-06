from Config import config
from train import train
from test import testTools
import numpy
from DATA import WaterDataset
import Networks
from torchvision import transforms as T
from torch.utils.data import DataLoader
from matplotlib import pyplot
import os
from PIL import Image
import time
import utils



def predict_all():
    '''已有三个sub数据
    三个训练好的模型
    生成九个预测 （相互交叉）
    '''
    path_model='D:\\Graduation\\Graduation_work\\Graduation_init\\MODELS'
    models=['model35','model36','model37']
    subs=['train_sub1','train_sub2','train_sub3']#subs 和 models 对应
    n_ite=1#从1开始计数
    path_pre='G:\\Graguation\\train_result'


    transform_img=config.transform_img
    transform_label=config.transform_label


    #模型设置
    model=getattr(Networks,config.model)(config.input_band,1)#创立网络对象

    '''model1 预测sub1 sub2 sub3'''
    path_model=path_model+'\\'+models[0]
    #加载模型
    model.load(path_model)
    if config.use_gpu:#使用gpu
        model=model.cuda()

    #sub1
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub1_model1'
    test_data=WaterDataset(sub=subs[0],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)

    #sub2
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub2_model1'
    test_data=WaterDataset(sub=subs[1],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)

    #sub3
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub3_model1'
    test_data=WaterDataset(sub=subs[2],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)


    '''model2 预测sub1 sub2 sub3'''
    path_model=path_model+'\\'+models[1]
    #加载模型
    model.load(path_model)
    if config.use_gpu:#使用gpu
        model=model.cuda()

    #sub1
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub1_model2'
    test_data=WaterDataset(sub=subs[0],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)

    #sub2
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub2_model2'
    test_data=WaterDataset(sub=subs[1],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)

    #sub3
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub3_model2'
    test_data=WaterDataset(sub=subs[2],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)

    '''model3 预测sub1 sub2 sub3'''
    path_model=path_model+'\\'+models[2]
    #加载模型
    model.load(path_model)
    if config.use_gpu:#使用gpu
        model=model.cuda()

    #sub1
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub1_model3'
    test_data=WaterDataset(sub=subs[0],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)

    #sub2
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub2_model3'
    test_data=WaterDataset(sub=subs[1],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)

    #sub3
    path_predict=path_pre+'\\'+str(n_ite)+'ite'+'\\'+'predict_'+str(n_ite)+'_sub3_model3'
    test_data=WaterDataset(sub=subs[2],train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)
    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)
    t=testTools(model=model,data=test_dataloader)
    t.predict(path_predict=path_predict)

def segment_merge():
    '''分割+合并
    分割  用一定的阈值
    合并 赋予每个像素 标签值相加/模型数  比如，有三个模型，某像素有一个模型认为时水（1），另外两个认为是非水体（0），则赋予  （0+0+1）/3
    '''

    n_ite=1
    path='G:\\Graguation\\train_result\\'+str(n_ite)+'ite'
    thre=[0.31,0.24,0.34]#对应 model1 2 3


    '''sub1'''
    name=['predict_'+str(n_ite)+'_sub1_model1' , 'predict_'+str(n_ite)+'_sub1_model2' , 'predict_'+str(n_ite)+'_sub1_model3']
    name_to='predict_'+str(n_ite)+'_sub1_merge'
    d=os.listdir(path+'\\'+name[0])
    for i in range(len(d)):
        if i%100==0:
            print(i)

        predict1=numpy.array(Image.open(path+'\\'+name[0]+'\\'+d[i]))
        predict2=numpy.array(Image.open(path+'\\'+name[1]+'\\'+d[i]))
        predict3=numpy.array(Image.open(path+'\\'+name[2]+'\\'+d[i]))

        ma=predict1.max()
        mi=predict1.min()
        t=thre[0]*(ma-mi)+mi
        img1=numpy.where(predict1>t,1,0)

        ma=predict2.max()
        mi=predict2.min()
        t=thre[1]*(ma-mi)+mi
        img2=numpy.where(predict2>t,1,0)

        ma=predict3.max()
        mi=predict3.min()
        t=thre[2]*(ma-mi)+mi
        img3=numpy.where(predict3>t,1,0)

        img=(img1+img2+img3)/3
        Image.fromarray((img*255).astype(numpy.uint8)).save(path+'\\'+name_to+'\\'+d[i])

    '''sub2'''
    name=['predict_'+str(n_ite)+'_sub2_model1' , 'predict_'+str(n_ite)+'_sub2_model2' , 'predict_'+str(n_ite)+'_sub2_model3']
    name_to='predict_'+str(n_ite)+'_sub2_merge'
    d=os.listdir(path+'\\'+name[0])
    for i in range(len(d)):
        if i%100==0:
            print(i)
        predict1=numpy.array(Image.open(path+'\\'+name[0]+'\\'+d[i]))
        predict2=numpy.array(Image.open(path+'\\'+name[1]+'\\'+d[i]))
        predict3=numpy.array(Image.open(path+'\\'+name[2]+'\\'+d[i]))

        ma=predict1.max()
        mi=predict1.min()
        t=thre[0]*(ma-mi)+mi
        img1=numpy.where(predict1>t,1,0)

        ma=predict2.max()
        mi=predict2.min()
        t=thre[1]*(ma-mi)+mi
        img2=numpy.where(predict2>t,1,0)

        ma=predict3.max()
        mi=predict3.min()
        t=thre[2]*(ma-mi)+mi
        img3=numpy.where(predict3>t,1,0)

        img=(img1+img2+img3)/3
        Image.fromarray((img*255).astype(numpy.uint8)).save(path+'\\'+name_to+'\\'+d[i])

    '''sub3'''
    name=['predict_'+str(n_ite)+'_sub3_model1' , 'predict_'+str(n_ite)+'_sub3_model2' , 'predict_'+str(n_ite)+'_sub3_model3']
    name_to='predict_'+str(n_ite)+'_sub3_merge'
    d=os.listdir(path+'\\'+name[0])
    for i in range(len(d)):
        if i%100==0:
            print(i)
        predict1=numpy.array(Image.open(path+'\\'+name[0]+'\\'+d[i]))
        predict2=numpy.array(Image.open(path+'\\'+name[1]+'\\'+d[i]))
        predict3=numpy.array(Image.open(path+'\\'+name[2]+'\\'+d[i]))

        ma=predict1.max()
        mi=predict1.min()
        t=thre[0]*(ma-mi)+mi
        img1=numpy.where(predict1>t,1,0)

        ma=predict2.max()
        mi=predict2.min()
        t=thre[1]*(ma-mi)+mi
        img2=numpy.where(predict2>t,1,0)

        ma=predict3.max()
        mi=predict3.min()
        t=thre[2]*(ma-mi)+mi
        img3=numpy.where(predict3>t,1,0)

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
    utils.random_label(path+'\\'+name_from[0],path+'\\'+name_to[0])

    '''sub2'''
    utils.random_label(path+'\\'+name_from[1],path+'\\'+name_to[1])

    '''sub3'''
    utils.random_label(path+'\\'+name_from[2],path+'\\'+name_to[2])

def masks():
    '''对utils里的mask 的集成
    '''
    n_ite=1
    path='G:\\Graguation\\train_result\\'+str(n_ite)+'ite'
    name_from=['predict_'+str(n_ite)+'_sub1_merge','predict_'+str(n_ite)+'_sub2_merge','predict_'+str(n_ite)+'_sub3_merge']
    name_to=['predict_'+str(n_ite)+'_sub1_mask','predict_'+str(n_ite)+'_sub2_mask','predict_'+str(n_ite)+'_sub3_mask']

    '''sub1'''
    utils.mask(path+'\\'+name_from[0],path+'\\'+name_to[0])

    '''sub2'''
    utils.mask(path+'\\'+name_from[1],path+'\\'+name_to[1])

    '''sub3'''
    utils.mask(path+'\\'+name_from[2],path+'\\'+name_to[2])

def merge_mask():
    '''utils 里的 masked_label 的集成
    '''
    n_ite=1
    path='G:\\Graguation\\train_result\\'+str(n_ite)+'ite'
    name_from=['predict_'+str(n_ite)+'_sub1_merge','predict_'+str(n_ite)+'_sub2_merge','predict_'+str(n_ite)+'_sub3_merge']
    name_mask=['predict_'+str(n_ite)+'_sub1_mask','predict_'+str(n_ite)+'_sub2_mask','predict_'+str(n_ite)+'_sub3_mask']
    name_to=['predict_'+str(n_ite)+'_sub1_merge_mask','predict_'+str(n_ite)+'_sub2_merge_mask','predict_'+str(n_ite)+'_sub3_merge_mask']

    '''sub1'''
    utils.masked_label(path+'\\'+name_from[0],path+'\\'+name_mask[0],path+'\\'+name_to[0])

    '''sub2'''
    utils.masked_label(path+'\\'+name_from[1],path+'\\'+name_mask[1],path+'\\'+name_to[1])

    '''sub3'''
    utils.masked_label(path+'\\'+name_from[2],path+'\\'+name_mask[2],path+'\\'+name_to[2])


def train_test():
    train()
    '''模型设置'''
    model=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    #加载模型
    model.load(config.model_save_path)
    if config.use_gpu:#使用gpu
        model=model.cuda()

    '''数据加载'''
    transform_img=config.transform_img
    transform_label=config.transform_label
    test_data=WaterDataset(train=False,val=False,transforms_img=transform_img,transforms_label=transform_label)


    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)



    t=testTools(model=model,data=test_dataloader)
    #t.predict()
    #t.pr(save=save,show=show)
    #t.predict(0.24)


    #t.getlabel_low_trained(0.48)
    #getlabel_low()


    p,r,f=t.pr_fmeasure()
    pa=t.PA()
    iou=t.IoU()

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
    t.recordPrecision(thre/100,p[thre],r[thre],f[thre],pa[thre],iou[thre])


    pyplot.plot(f,label='f-measure')
    pyplot.plot(pa,label='pa')
    pyplot.plot(iou,label='iou')
    pyplot.legend()
    pyplot.show()

    t.drawStatistic()



if __name__=='__main__':
    #segment_merge()
    #random_labels()
    #masks()
    train_test()
