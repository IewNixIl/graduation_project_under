from skimage import exposure
from skimage import io
from matplotlib import pyplot
import shelve
import os
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
    with shelve.open(config.path_devision) as f:
        tr=f['train_water']
        
        print(len(tr))
        

