'''将数据集划分为训练数据、测试数据'''
import os
import shelve
import random
from skimage import io
import numpy
import shutil
import sys
sys.path.append('..')
from Config import config


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

    
    return path_lc


def devide(path,path_save):
    '''划分为训练、测试
    共180000左右的数据 从中随机30000作为测试数据
    用列表记录训练、测试  [[0,4,33],...] 
                        第一位表示季节，0:spring；1：summer。。。。。
                        第二位第三位为具体图像标号
                        [0,4,33] 表示 ROIs1158_spring_lc_4_p33
    path 所有数据的文件夹
    path_save 列表保存的文件路径
    '''
    all=[]
    dir1=os.listdir(path)
    for i in dir1:
        if 'spring' in i:
            season=0
        elif 'summer' in i:
            season=1
        elif 'fall' in i:
            season=2
        elif 'winter' in i:
            season=3
        dir2=os.listdir(path+'\\'+i)
        for j in dir2:
            
            if 'lc' in j:
                first=int(j.split('_')[-1])
                dir3=os.listdir(path+'\\'+i+'\\'+j)
                for k in dir3:
                    last=int(k.split('_')[-1].split('.')[0][1:])
                    all.append([season,first,last])
    
    all=numpy.array(all)
    flag=random.sample(range(all.shape[0]),30000)
    test_list=all[flag,:].tolist()
    train_list=all[list(set(range(all.shape[0]))-set(flag)),:].tolist()
    
    with shelve.open(path_save) as f:
        f['train']=train_list
        f['test']=test_list
    
    
def statistic(path_save):
    '''统计训练数据和测试数据各个季节的数量
    '''                
    train=numpy.zeros(4)
    test=numpy.zeros(4)
    train_sub=numpy.zeros(4)
    with shelve.open(path_save) as f:
        #tr=f['train']
        #te=f['test']
        tr_sub=f['train']
        '''
        for i in tr:
            if i[0]==0:
                train[0]+=1
            if i[0]==1:
                train[1]+=1
            if i[0]==2:
                train[2]+=1
            if i[0]==3:
                train[3]+=1
        for i in te:
            if i[0]==0:
                test[0]+=1
            if i[0]==1:
                test[1]+=1
            if i[0]==2:
                test[2]+=1
            if i[0]==3:
                test[3]+=1
        '''
        for i in tr_sub:
            if i[0]==0:
                train_sub[0]+=1
            if i[0]==1:
                train_sub[1]+=1
            if i[0]==2:
                train_sub[2]+=1
            if i[0]==3:
                train_sub[3]+=1
    '''
    print(train)
    print(test)
    '''
    print(train_sub)
    '''
    print(test/train)
    print(train_sub/train)
    '''
    
def sub():
    '''训练集中抽取4000小的训练集
    '''
    
    sub1=[]
    sub2=[]
    sub3=[]
    
    with shelve.open(config.path_devision) as f:
        flag=random.sample(range(len(f['train_water'])),15000)
        t=numpy.array(f['train_water'])
        tt=t[flag,:].tolist()
        sub1=tt[:5000]
        sub2=tt[5000:10000]
        sub3=tt[10000:15000]
        flag=random.sample(range(len(f['train_nowater'])),15000)
        t=numpy.array(f['train_nowater'])
        tt=t[flag,:].tolist()
        sub1=sub1+tt[:5000]
        sub2=sub2+tt[5000:10000]
        sub3=sub3+tt[10000:15000]
        
        f['train_sub1']=sub1
        f['train_sub2']=sub2
        f['train_sub3']=sub3
        
        
        
def val_test(rate):
    '''将所有手动标记的labels 分为测试集、验证集，并复制到相应的文件夹中
    rate是指 验证集数量/总数量
    '''
    d=os.listdir(config.path_labels)
    n_val=int(len(d)*rate)
    n_test=len(d)-n_val
    flag_all=set(range(len(d)))
    flag_val=set(random.sample(range(len(d)),n_val))
    flag_test=flag_all-flag_val
    
    for i in range(len(d)):
        if i in flag_test:
            shutil.copyfile(config.path_labels+'\\'+d[i],config.path_labels_test+'\\'+d[i])
        elif i in flag_val:
            shutil.copyfile(config.path_labels+'\\'+d[i],config.path_labels_val+'\\'+d[i])
    print('val:'+str(n_val))
    print('test:'+str(n_test))
    
def water_nowater():
    water=[]
    nowater=[]
    with shelve.open(config.path_devision) as f:
        tr=f['train']
        n=0
        for i in range(len(tr)):
            p=getname(config.path,tr[i])
            label=io.imread(p)[:,:,0]
            if (label==17).any():
                water.append(tr[i])
            else:
                nowater.append(tr[i])
            if i%100==0:
                print(i)
        f['train_water']=water
        f['train_nowater']=nowater
    
    
if __name__=='__main__':
    

    
    #with shelve.open(config.path_devision) as f:
    #    for i in f:
    #        print(i+'  '+str(len(f[i])))
            
        
    
    statistic(config.path_devision)
    #path=config.path
    #path_save=config.path_devision
    #val_test(0.2)
    #devide(path,path_save)
    
    #statistic(path_save)
    #sub()     
    #with shelve.open(path_save) as f:
    #    for i in f:
    #        print(i)
       
    
    

