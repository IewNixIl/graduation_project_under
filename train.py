from DATA import WaterDataset
import Networks
from Config import config
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
import time
import datetime
import shelve
import numpy
import random
from test import testTools
import copy


def train(model_path=config.model_save_path,statistic_path=config.statistic_save_path):

    '''模型设置'''

    model=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    if config.model_load_path:#加载模型
        model.load(config.model_load_path)

    if config.use_gpu:#使用gpu
        model=model.cuda()

    '''数据加载'''
    transform_img=config.transform_img
    transform_label=config.transform_label
    train_data=WaterDataset(train=True,val=False,transforms_img=transform_img,transforms_label=transform_label)


    train_dataloader=DataLoader(train_data,config.batch_size,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)

    test_data=WaterDataset(train=False,val=True,transforms_img=transform_img,transforms_label=transform_label)


    test_dataloader=DataLoader(test_data,1,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=1)

    print('data loaded')
    '''目标函数和优化器'''
    criterion=torch.nn.BCELoss()
    learning_rate=config.learning_rate
    optimizer=torch.optim.Adam(model.parameters(),
                            lr=learning_rate,
                            weight_decay=config.weight_decay)

    '''测试工具'''
    testtools=testTools(model=model,data=test_dataloader)
    print('testtools prepared!')


    '''记录、统计'''
    recording_loss=[]
    running_loss=0#中间数据
    recording_iou=[]
    iou_max_pre=0#记录之前最大的iou
    checkpoint=None
    number_unchange=0


    '''训练'''

    for epoch in range(config.max_epoch):
        t1=time.time()
        for i,data in enumerate(train_dataloader,0):
            if config.path_mask_train:
                inputs,labels,name,mask=data#获得输入和标签
                if config.use_gpu:#使用gpu
                    inputs=inputs.cuda()
                    labels=labels.cuda()
                    mask=mask.cuda()
            else:
                inputs,labels,name=data#获得输入和标签
                if config.use_gpu:#使用gpu
                    inputs=inputs.cuda()
                    labels=labels.cuda()

            optimizer.zero_grad()#积累的导数归零


            if config.path_mask_train:
                outputs=model(inputs)*mask#前向传播
            else:
                outputs=model(inputs)

            loss=criterion(outputs,labels)#计算损失

            #记录
            running_loss+=loss.item()





            if i % config.number_recording == config.number_recording - 1:
                recording_loss.append(running_loss/config.number_recording)



                iou=testtools.IoU(recaculate=True).max()
                recording_iou.append(iou)


                number_unchange+=1
                if iou>iou_max_pre:
                    iou_max_pre=iou
                    torch.save(model,config.model_save_path_checkpoints)
                    number_unchange=0

                running_loss=0

                print('epoch:'+str(epoch)+',batches:'+str(i-config.number_recording+1)+'--'+str(i)+
                ',loss:'+str(recording_loss[-1])+', max_iou:'+str(iou_max_pre))


            loss.backward()#后向传播
            optimizer.step()#更新参数
        t2=time.time()
        print('time: '+str((t2-t1)/60)+' min')

    model_dict = torch.load(config.model_save_path_checkpoints).state_dict()
    # 载入参数
    model.load_state_dict(model_dict)



    '''保存模型'''
    #if config.model_save_path:
    #    model.save(config.model_save_path)

    model.save(model_path)

    '''保存统计数据'''
    #if config.statistic_save_path:
    #    with shelve.open(config.statistic_save_path) as f:
    #        f['loss']=recording_loss
    #        f['iou']=recording_iou
    with shelve.open(statistic_path) as f:
        f['loss']=recording_loss
        f['iou']=recording_iou


    print('Finished Training!')




if __name__=='__main__':
    train()
