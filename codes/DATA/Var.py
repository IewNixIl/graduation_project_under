from skimage import io
import numpy
from matplotlib import pyplot
import os
from PIL import Image



if __name__=='__main__':
    path_predict='E:\\predict'
    path_down='E:\\predict_downsample'
    
    d=os.listdir(path_predict)
    for i in range(len(d)):
        Image.open(path_predict+'\\'+d[i]).resize((16,16)).resize((256,256)).save(path_down+'\\'+d[i])
        
        
        
    
    