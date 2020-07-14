from skimage import exposure
from skimage import io
from matplotlib import pyplot
import shelve
import os
import numpy
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

if __name__=='__main__':
    models=['model25','model26','model27']
    #models=['model1','model7','model10','model13','model16','model2','model8','model11','model14','model17','model3','model9','model12','model15','model18']
    r1=[]
    r2=[]
    r3=[]
    with shelve.open(config.path_acc_record) as f:

        for i in range(1):
            r1.append(f[models[i]])
            r2.append(f[models[i+1]])
            r3.append(f[models[i+2]])

    r1=numpy.array(r1)
    r2=numpy.array(r2)
    r3=numpy.array(r3)
    print(r1)
    print(r2)
    print(r3)
    '''
    n=5
    pyplot.plot(r1[:,n])
    pyplot.plot(r2[:,n])
    pyplot.plot(r3[:,n])
    print(r1)
    print(r2)
    print(r3)
    pyplot.show()
    '''
