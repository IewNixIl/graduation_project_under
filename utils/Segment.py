from skimage import io
from matplotlib import pyplot
from skimage import io,exposure
from skimage.segmentation import slic,mark_boundaries
import numpy
from sklearn.cluster import KMeans,AgglomerativeClustering,MiniBatchKMeans
from collections import Counter
from PIL import Image
import time
import random
import os
import sys
sys.path.append('..')
from Config import config


def correct_merge(path_from,path_to):
    d=os.listdir(path_from)
    for i in range(len(d)):

        correct_merge_single(path_from+'\\'+d[i],path_to+'\\'+d[i])
        if i%100==0:
            print(i)

def correct_merge_single(path_merge,path_to):
    '''输入一张merge后的label的路径
    利用分割进行修正，对不确定像素进行改正
    返回1-0 图片
    path_merge merge后单张label的路径
    '''

    #t=[]
    #t.append(time.time())

    n_segments=500
    compactness=20

    merge=io.imread(path_merge)
    result=merge.copy()
    namelist=transform(path_merge.split('\\')[-1])
    path_s1,path_s2=getname(config.path,namelist)
    s2=(io.imread(path_s2)[:,:,[1,2,3,4,5,6,7,8,11,12]].astype(numpy.float32)-2048)/2048
    s1=(io.imread(path_s1)+11)/11
    img=numpy.zeros((s1.shape[0],s1.shape[1],s1.shape[2]+s2.shape[2])).astype(numpy.float32)
    img[:,:,:2]=s1
    img[:,:,2:]=s2


    #img=((img-img.min())/(img.max()-img.min())*255).astype(numpy.int8)

    #t.append(time.time())

    #segment=slic(img,n_segments=n_segments, compactness=compactness,enforce_connectivity=True,max_iter=7)

    #t.append(time.time())

    img.shape=-1,img.shape[2]
    #estimator = KMeans(n_clusters=10,n_init=2)#构造聚类器
    estimator = MiniBatchKMeans(n_clusters=13,batch_size=10000)
    #estimator=AgglomerativeClustering(n_clusters=10,affinity='euclidean',linkage='complete')
    estimator.fit(img)#聚类

    segment = estimator.labels_
    segment.shape=merge.shape



    mask=numpy.where(merge==0,False,True)*numpy.where(merge==255,False,True)
    dd=Counter(segment[mask])

    #t.append(time.time())

    for i in dd:
        flag=numpy.where(segment==i,True,False)
        d=Counter(merge[flag])


        result[flag*mask]=max(d,key=d.get)


    #t.append(time.time())

    Image.fromarray(result).save(path_to)

    #t.append(time.time())

    #for i in range(1,len(t)):
    #    print((t[i]-t[i-1])/(t[-1]-t[0]),end=' ')
    #print()




def getname(path,namelist):
    if namelist[0]==0:
        season='ROIs1158_spring'
    elif namelist[0]==1:
        season='ROIs1868_summer'
    elif namelist[0]==2:
        season='ROIs1970_fall'
    elif namelist[0]==3:
        season='ROIs2017_winter'

    path_s2=path+'\\'+season+'\\s2_'+str(namelist[1])+'\\'+season+'_s2_'+str(namelist[1])+'_p'+str(namelist[2])+'.tif'
    path_s1=path+'\\'+season+'\\s1_'+str(namelist[1])+'\\'+season+'_s1_'+str(namelist[1])+'_p'+str(namelist[2])+'.tif'

    return path_s1,path_s2

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


if __name__=='__main__':

    t1=time.time()
    correct_merge('D:\\result\\rest_result\\1ite\\merge','D:\\result\\rest_result\\1ite\\merge_correct')
    t2=time.time()
    print(t2-t1)
