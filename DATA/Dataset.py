from torch.utils import data
from skimage import io,filters
from skimage.morphology import disk
from PIL import Image
import numpy
from matplotlib import pyplot
import shelve
import os
import sys
sys.path.append('..')
from Config import config

class WaterDataset(data.Dataset):
    def __init__(self,train=True,transforms_img=None,transforms_label=None,val=False,sub=config.sub_dataset_train,path_mask=config.path_mask_train,path_label=config.path_label_train):
        if train:
            with shelve.open(config.path_devision) as f:
                self.devision=f[sub]

            if path_label:
                self.path_labels=path_label
            else:
                self.path_labels=''

        else:
            if val:
                self.devision=self.getlist(config.path_labels_val)
                self.path_labels=config.path_labels_val
            else:
                self.devision=self.getlist(config.path_labels_test)
                self.path_labels=config.path_labels_test
        self.path_mask=path_mask
        self.transforms_img=transforms_img
        self.transforms_label=transforms_label
        self.iftrain=train
        self.ifval=val




    def __len__(self):
        return len(self.devision)

    def __getitem__(self,index):
        #name_lc,name_s1,name_s2=getname(config.path,self.devision[index])
        if not self.path_mask:
            name_label,name_s1,name_s2=self.decode(index)
        else:
            name_label,name_s1,name_s2,name_mask=self.decode(index)
            mask=self.readmask(name_mask)

        img=self.readimg(name_s1,name_s2)
        label=self.readlabel(name_label)

        #print(label.max())


        #label=numpy.array(Image.fromarray(label).resize((16,16)))
        #print(label.shape)
        #print(label.max())




        if self.transforms_img:
            img=self.transforms_img(img)
        if self.transforms_label:
            label=self.transforms_label(label)
            if self.path_mask:
                mask=self.transforms_label(mask)


        name=name_label.split('\\')[-1]
        name=name.split('_')
        name[2]='predict'
        name='_'.join(name)

        if self.path_mask:
            return img,label,name,mask
        else:
            return img,label,name

    def getlist(self,path):
        dirlist=os.listdir(path)
        result=[]
        for i in dirlist:
            d=[]
            if 'spring' in i:
                d.append(0)
            elif 'summer' in i:
                d.append(1)
            elif 'fall' in i:
                d.append(2)
            elif 'winter' in i:
                d.append(3)

            d.append(int(i.split('_')[3]))

            d.append(int(i.split('_')[4].split('.')[0][1:]))

            result.append(d)

        return result

    def decode(self,index):
        path=config.path

        if self.devision[index][0]==0:
            season='ROIs1158_spring'
        elif self.devision[index][0]==1:
            season='ROIs1868_summer'
        elif self.devision[index][0]==2:
            season='ROIs1970_fall'
        elif self.devision[index][0]==3:
            season='ROIs2017_winter'

        path_s1=path+'\\'+season+'\\s1_'+str(self.devision[index][1])+'\\'+season+'_s1_'+str(self.devision[index][1])+'_p'+str(self.devision[index][2])+'.tif'
        path_s2=path+'\\'+season+'\\s2_'+str(self.devision[index][1])+'\\'+season+'_s2_'+str(self.devision[index][1])+'_p'+str(self.devision[index][2])+'.tif'


        if not self.path_labels:
            path_lc=path+'\\'+season+'\\lc_'+str(self.devision[index][1])+'\\'+season+'_lc_'+str(self.devision[index][1])+'_p'+str(self.devision[index][2])+'.tif'
        else:
            path_lc=self.path_labels+'\\'+season+'_label_'+str(self.devision[index][1])+'_p'+str(self.devision[index][2])+'.tif'


        if not self.path_mask:
            return path_lc,path_s1,path_s2
        else:
            path_m=self.path_mask+'\\'+season+'_label_'+str(self.devision[index][1])+'_p'+str(self.devision[index][2])+'.tif'
            return path_lc,path_s1,path_s2,path_m


    def readimg(self,path_s1,path_s2):
        s1=io.imread(path_s1)
        s2=io.imread(path_s2)[:,:,[1,2,3,4,5,6,7,8,11,12]]
        if config.use_denoise:
            s1[:,:,0]=filters.median(s1[:,:,0],disk(5))
            s1[:,:,1]=filters.median(s1[:,:,1],disk(5))


        if config.input_band==2:
            img=s1
        elif config.input_band==10:
            img=s2
        elif config.input_band==12:
            img=numpy.zeros((s1.shape[0],s1.shape[1],s1.shape[2]+s2.shape[2]))
            img[:,:,:2]=s1
            img[:,:,2:]=s2
        elif config.input_band==13:
            img=numpy.zeros((s1.shape[0],s1.shape[1],s1.shape[2]+s2.shape[2]+1))
            img[:,:,:2]=s1
            img[:,:,2:-1]=s2
            t=(s2[:,:,6]-s2[:,:,1])/(s2[:,:,6]+s2[:,:,1])
            t[numpy.isnan(t)]=0
            t[numpy.isinf(t)]=0
            img[:,:,-1]=t


        return img.astype(numpy.float32)

    def readlabel(self,path_lc):
        if not self.path_labels:
            lc=io.imread(path_lc)[:,:,0]
            label=numpy.where(lc==17,1,0)
        else:
            label=io.imread(path_lc)/255


        return label.astype(numpy.float32)

    def readmask(self,path_m):
        mask=io.imread(path_m)
        return mask.astype(numpy.float32)


'''
def readimg(path_s1,path_s2):
    s1=io.imread(path_s1)
    s2=io.imread(path_s2)[:,:,[1,2,3,4,5,6,7,8,11,12]]
    if config.use_denoise:
        s1[:,:,0]=filters.median(s1[:,:,0],disk(5))
        s1[:,:,1]=filters.median(s1[:,:,1],disk(5))


    if config.input_band==2:
        img=s1
    elif config.input_band==10:
        img=s2
    elif config.input_band==12:
        img=numpy.zeros((s1.shape[0],s1.shape[1],s1.shape[2]+s2.shape[2]))
        img[:,:,:2]=s1
        img[:,:,2:]=s2

    return img.astype(numpy.float32)
'''
'''
def readlabel(path_lc,iforigin):
    #'''#iforigin 是否使用原始标签
    #'''
'''
    if iforigin:
        lc=io.imread(path_lc)[:,:,0]
        label=numpy.where(lc==17,1,0)
    else:
        label=io.imread(path_lc)/255


    return label.astype(numpy.float32)

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
    path_s1=path+'\\'+season+'\\s1_'+str(namelist[1])+'\\'+season+'_s1_'+str(namelist[1])+'_p'+str(namelist[2])+'.tif'
    path_s2=path+'\\'+season+'\\s2_'+str(namelist[1])+'\\'+season+'_s2_'+str(namelist[1])+'_p'+str(namelist[2])+'.tif'

    return path_lc,path_s1,path_s2


def getlabellist(path):

    dirlist=os.listdir(path)
    result=[]
    for i in dirlist:
        lc.append(path+'\\'+i)
        d=[]
        if 'spring' in i:
            d.append(0)
        elif 'summer' in i:
            d.append(1)
        elif 'fall' in i:
            d.append(2)
        elif 'winter' in i:
            d.append(3)

        d.append(int(i.split('_')[3]))

        d.append(int(i.split('_')[4].split('.')[0][1:]))

        result.append(d)

    return result
'''

if __name__=='__main__':

    train=True
    val=False
    transform_img=config.transform_img
    transform_label=config.transform_label
    path_model='D:\\codes\\Graduation\\MODELS\\model1'
    path_sta='D:\\codes\\Graduation\\MODELS\\sta1'
    path_label=''
    path_mask=''
    sub_name='train_sub1'
    train_data=WaterDataset(train=train,val=val,transforms_img=transform_img,transforms_label=transform_label,sub=sub_name,path_label=path_label,path_mask=path_mask)
    #img,label=train_data[1003]
    print(len(train_data))
    for i in range(len(train_data)):
        img,label,name=train_data[i]
        mask=label.numpy()
        label=(mask[0,:,:]*255).astype(int)
        if label.max()==0:
            continue
        print(label.max())
        print(label.min())
        pyplot.imshow(label,cmap='gray')
        pyplot.show()
