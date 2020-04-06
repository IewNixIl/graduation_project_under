from DATA import WaterDataset
import Networks
from Config import config
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
import time
import os
import numpy
from PIL import Image
from matplotlib import pyplot
from skimage import io,exposure
import shelve



class testTools:
    '''测试工具
    '''
    def __init__(self,model,data):
        self.model=model


        self.test_dataloader=data

        self.have_cacu_confu=False


    def predict(self,path_predict=config.path_predict_save):
        '''生成每张图像的预测图
        不进行阈值分割
        '''

        for i,data in enumerate(self.test_dataloader,0):
            if i%100==0:
                print(i)
            inputs,labels,name=data#获得输入和标签
            if config.use_gpu:#使用gpu
                inputs=inputs.cuda()

            outputs=self.model(inputs)

            predict=numpy.array(outputs[0,0,:,:].cpu().detach())

            Image.fromarray(predict).save(path_predict+'\\'+name[0])

    def confusion(self):
        confu=numpy.zeros((2,2,100))
        thre=numpy.arange(0,100)/100
        for i,data in enumerate(self.test_dataloader,0):
            #print(i)
            inputs,labels,name=data#获得输入和标签

            if config.use_gpu:#使用gpu
                inputs=inputs.cuda()
                labels=labels.cuda()

            outputs=self.model(inputs)
            #ma=outputs.max().item()
            #mi=outputs.min().item()


            for j in range(100):

                #predict=torch.where(outputs>torch.tensor([thre[j]*(ma-mi)+mi]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda())
                predict=torch.where(outputs>torch.tensor([thre[j]]).cuda(),torch.tensor([1]).cuda(),torch.tensor([0]).cuda())
                label_true=labels==1
                predict_true=predict==1
                label_false=labels==0
                predict_false=predict==0
                TP=torch.sum(label_true*predict_true).item()
                FN=torch.sum(label_true*predict_false).item()
                FP=torch.sum(label_false*predict_true).item()
                TN=torch.sum(label_false*predict_false).item()



                confu[0,0,j]=confu[0,0,j]+TP
                confu[0,1,j]=confu[0,1,j]+FN
                confu[1,0,j]=confu[1,0,j]+FP
                confu[1,1,j]=confu[1,1,j]+TN
        self.confu=confu
        self.have_cacu_confu=True

    def pr_fmeasure(self,recaculate=False):
        '''生成pr曲线
        '''
        if recaculate or not self.have_cacu_confu:
            self.confusion()
        confu=self.confu


        '''测试'''

        #r=(self.confu[0,0,:]+self.confu[1,0,:])!=0
        #confu=confu[:,:,r]
        p=confu[0,0,:]/(confu[0,0,:]+confu[1,0,:])
        r=confu[0,0,:]/(confu[0,0,:]+confu[0,1,:])
        fmeasure=2*p*r/(p+r)

        '''
        pl=p
        rl=r
        arg=numpy.argsort(rl).tolist()
        pl=pl[arg]
        rl=rl[arg]
        '''

        return p,r,fmeasure

    def PA(self,recaculate=False):
        '''pixel accuracy
        分类正确的像素数（不论哪一类）/总像素数
        '''

        if recaculate or not self.have_cacu_confu:
            self.confusion()
        confu=self.confu


        pa=(confu[0,0,:]+confu[1,1,:])/(confu.sum(axis=0).sum(axis=0))



        return pa

    def MPA(self,recaculate=False):
        '''mean pixel accuracy
        对于标签的每一类像素中分类正确的比例的平均值
        是按类计算然后平均
        '''
        if recaculate or not self.have_cacu_confu:
            self.confusion()
        confu=self.confu


        mpa=confu[0,0]/(confu[0,0]+confu[0,1])+confu[1,1]/(confu[1,0]+confu[1,1])
        mpa=mpa/2


        return mpa

    def IoU(self,recaculate=False):
        '''intersection over union
        '''
        if recaculate or not self.have_cacu_confu:
            self.confusion()
        confu=self.confu

        iou=confu[0,0]/(confu[0,0]+confu[0,1]+confu[1,0])


        return iou

    def MIoU(self,recaculate=False):
        '''mean intersection over union
        也是对类计算然后平均
        '''
        if recaculate or not self.have_cacu_confu:
            self.confusion()
        confu=self.confu


        miou=confu[0,0]/(confu[0,0]+confu[0,1]+confu[1,0])+confu[1,1]/(confu[1,0]+confu[1,1]+confu[0,1])
        miou=miou/2


        return miou

    def recordPrecision(self,thre,p,r,fmeasure,pa,iou):
        '''记录精度指标
        '''
        name=config.model_save_path.split('\\')[-1]
        with shelve.open(config.path_acc_record) as f:
            f[name]=[thre,p,r,fmeasure,pa,iou]


    def drawStatistic(self):
        with shelve.open(config.statistic_save_path) as f:
            loss=f['loss']
            iou=f['iou']

        pyplot.subplot(1,2,1)
        pyplot.plot(loss)
        pyplot.subplot(1,2,2)
        pyplot.plot(iou)
        pyplot.show()

    def integrate(self,path,models):
        '''用多个模型的预测的交集作为预测结果的测试程序
        path 存模型的文件夹路径
        models  模型名称的列表 比如:[model1,model2,model3]
        返回 fmeasure pa iou
        '''
        m=[]
        for i in models:
            model=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
            #加载模型
            model.load(path+'\\'+i)
            if config.use_gpu:#使用gpu
                model=model.cuda()
            m.append(model)


        confu=numpy.zeros((2,2,100))
        thre=numpy.arange(0,100)/100
        for i,data in enumerate(self.test_dataloader,0):

            print(i)
            inputs,labels,name=data#获得输入和标签

            if config.use_gpu:#使用gpu
                inputs=inputs.cuda()
                labels=labels.cuda()

            outputs=[]

            for j in m:
                outputs.append(j(inputs)[0])


            predict=torch.zeros(outputs[0].size()).cuda()
            for j in range(100):
                for k in outputs:
                    ma=k.max()
                    mi=k.min()
                    predict=predict+torch.where(k>torch.tensor([thre[j]*(ma-mi)+mi]).cuda(),torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())
                #predict=torch.where(predict>torch.tensor([len(outputs)/2]).cuda(),torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())
                predict=torch.where(predict>torch.tensor([0.5]).cuda(),torch.tensor([1.]).cuda(),torch.tensor([0.]).cuda())

                label_true=labels==1
                predict_true=predict==1
                label_false=labels==0
                predict_false=predict==0
                TP=torch.sum(label_true*predict_true).item()
                FN=torch.sum(label_true*predict_false).item()
                FP=torch.sum(label_false*predict_true).item()
                TN=torch.sum(label_false*predict_false).item()



                confu[0,0,j]=confu[0,0,j]+TP
                confu[0,1,j]=confu[0,1,j]+FN
                confu[1,0,j]=confu[1,0,j]+FP
                confu[1,1,j]=confu[1,1,j]+TN


        p=confu[0,0,:]/(confu[0,0,:]+confu[1,0,:])
        r=confu[0,0,:]/(confu[0,0,:]+confu[0,1,:])
        fmeasure=2*p*r/(p+r)
        pa=(confu[0,0,:]+confu[1,1,:])/(confu.sum(axis=0).sum(axis=0))
        iou=confu[0,0,:]/(confu[0,0,:]+confu[0,1,:]+confu[1,0,:])

        return p,r,fmeasure,pa,iou








def getname(path,namelist):
    if namelist[0]==0:
        season='ROIs1158_spring'
    elif namelist[0]==1:
        season='ROIs1868_summer'
    elif namelist[0]==2:
        season='ROIs1970_fall'
    elif namelist[0]==3:
        season='ROIs2017_winter'

    path_lc=path+'\\'+season+'\\lc_'+str(namelist[1])+'\\'+season+'_lc_'+str(namelist[1])+'_p'+str(namelist[2])+'.tif'
    path_s2=path+'\\'+season+'\\s2_'+str(namelist[1])+'\\'+season+'_s2_'+str(namelist[1])+'_p'+str(namelist[2])+'.tif'
    path_s1=path+'\\'+season+'\\s1_'+str(namelist[1])+'\\'+season+'_s1_'+str(namelist[1])+'_p'+str(namelist[2])+'.tif'

    return path_s2,path_lc,path_s1

def transform(name):
    if 'spring' in name:
        season=0
    elif 'summer' in name:
        season=1
    elif 'fall' in name:
        season=2
    elif 'winter' in name:
        season=3

    l=[]
    l.append(season)
    l.append(int(name.split('_')[3]))
    l.append(int(name.split('_')[4].split('.')[0][1:]))

    return l

def getimg():
    '''获得labels文件夹中原图
    '''
    dirlist=os.listdir(config.path_labels_test)
    for i in dirlist:
        l=transform(i)
        name,lc=getname(config.path,l)
        img=io.imread(name)[:,:,3:0:-1]
        img=(img-img.min())/(img.max()-img.min())
        img=img*255
        img=img.astype(numpy.uint8)

        img=exposure.adjust_gamma(img,0.5)

        Image.fromarray(img).save(config.path_img_save+'\\'+i)

def getsar(path_sar_1,path_sar_2):
    '''获得labels文件夹中原图
    '''
    dirlist=os.listdir(config.path_labels_test)
    for i in dirlist:
        l=transform(i)
        name,lc,sar=getname(config.path,l)
        img=io.imread(sar)

        img=(img-img.min())/(img.max()-img.min())
        img=img*255
        img=img.astype(numpy.uint8)

        #img=exposure.adjust_gamma(img,0.5)

        Image.fromarray(img[:,:,0]).save(path_sar_1+'\\'+i)
        Image.fromarray(img[:,:,1]).save(path_sar_2+'\\'+i)

def getlabel_low(path_to):
    '''获得labels文件夹中原图
    '''
    dirlist=os.listdir(config.path_labels_test)
    for i in dirlist:
        l=transform(i)
        name,lc,sar=getname(config.path,l)
        img=io.imread(lc)[:,:,0]

        img=numpy.where(img==17,255,0)
        img=img.astype(numpy.uint8)

        #img=exposure.adjust_gamma(img,0.5)

        Image.fromarray(img).save(path_to+'\\'+i)


def getndwi():
    '''获得labels文件夹中ndwi
    '''
    dirlist=os.listdir(config.path_labels)
    for i in dirlist:

        l=transform(i)
        name,lc=getname(config.path,l)
        img=io.imread(name)[:,:,[2,7]]
        img=(img[:,:,1]-img[:,:,0])/(img[:,:,1]+img[:,:,0])
        img=(img-img.min())/(img.max()-img.min())
        img=img*255
        img=img.astype(numpy.uint8)


        Image.fromarray(img).save(config.path_ndwi_save+'\\'+i)

def getlabel_train(threshold):
    '''保存用于训练的标签
    '''
    model=getattr(Networks,config.model)(config.input_band,1)#创立网络对象
    #加载模型
    model.load(config.model_save_path)
    if config.use_gpu:#使用gpu
        model=model.cuda()

    transform_img=config.transform_img
    transform_label=config.transform_label
    train_data=WaterDataset(transforms_img=transform_img,transforms_label=transform_label)


    train_dataloader=DataLoader(train_data,config.batch_size,
                            shuffle=True,#每一epoch打乱数据
                            num_workers=config.num_workers)

    for i,data in enumerate(train_dataloader,0):
        print(i)
        inputs,labels,name=data#获得输入和标签
        if config.use_gpu:#使用gpu
            inputs=inputs.cuda()

        outputs,ppp=self.model(inputs)

        predict=numpy.array(outputs[0,0,:,:].cpu().detach())
        predict=(predict-predict.min())/(predict.max()-predict.min())
        predict=numpy.where(predict>threshold,255,0)
        predict=predict.astype(numpy.uint8)

        Image.fromarray(predict).save(config.path_predict_save+'\\'+name[0])



if __name__=='__main__':
    '''
    getimg()
    getndwi()
    show=False
    save=False
    thre=

    t=testTools()
    t.predict(threshold=thre)
    '''


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

    #getsar('G:\\Graguation\\test_result\\sar1','G:\\Graguation\\test_result\\sar2')
    #getlabel_low('G:\\Graguation\\test_result\\lowlabel')
    t=testTools(model=model,data=test_dataloader)
    #t.predict()
    #t.pr(save=save,show=show)
    #t.predict(0.24)


    #t.getlabel_low_trained(0.48)
    #getlabel_low()
    t.drawStatistic()

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
