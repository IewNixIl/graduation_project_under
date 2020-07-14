import numpy
from matplotlib import pyplot
import gdal
from skimage import io,exposure
from skimage.segmentation import slic,mark_boundaries
import os
from PIL import Image
import shelve
import sys
sys.path.append('..')
from Config import config




def seg(path,n_segments=500, compactness=20):
    i=io.imread(path)[:,:,[3,2,1,7]]
    img=i[:,:,:3]
    img=(img-img.min())/(img.max()-img.min())
    img=img*255
    img=img.astype(numpy.uint8)

    img=exposure.adjust_gamma(img,0.5)

    


    
    return segment,out,img,wdi
    

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
    
    return path_s2
    
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
    

class UI:
    def __init__(self,mode='normal',init=0):
        '''mode = normal   正常
        mode=review  仅仅显示已经标记的 
        '''
        self.mode=mode
        
        self.path_merge='D:\\Codes\\test_dataset\\1ite\\predict_1_merge'
        self.path_mask='D:\\Codes\\test_dataset\\1ite\\predict_1_mask'
        self.path_label='D:\\Codes\\test_dataset\\label'
        self.path_random='D:\\Codes\\test_dataset\\1ite\\predict_1_random'
        self.path_smooth='D:\\Codes\\test_dataset\\1ite\\predict_1_merge_smooth_5'
        self.path_predict1='D:\\Codes\\test_dataset\\1ite\\predict_1_model1'
        self.path_predict2='D:\\Codes\\test_dataset\\1ite\\predict_1_model2'
        self.path_predict3='D:\\Codes\\test_dataset\\1ite\\predict_1_model3'
        
        self.n=init
        
        self.l=os.listdir(self.path_label)
        
        fig=pyplot.figure()
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.mpl_connect('key_press_event',self.on_key_press)
        
        self.fig=fig
        self.ax1=fig.add_subplot(3,1,1)
        self.ax2=fig.add_subplot(3,3,4)
        self.ax3=fig.add_subplot(3,3,5)
        self.ax4=fig.add_subplot(3,3,6)
        self.ax5=fig.add_subplot(3,1,3)
        #self.ax4=fig.add_subplot(2,2,4)

        pyplot.get_current_fig_manager().window.state('zoomed')
        #self.ax2=fig.add_subplot(1,2,2)
        
        

        self.draw()
        
        
        
        pyplot.show()
        
    def on_key_press(self,event):
        if event.key=='a' or event.key=='left':
            self.n-=1
            print(self.n)
            self.draw()

        if event.key=='d' or event.key=='right':
            if self.n+1>=len(self.l):
                return
            self.n+=1
        
            self.draw()


            
        

        
    def draw(self):
        

        print(self.l[self.n])
        
        self.label=io.imread(self.path_label+'\\'+self.l[self.n])
        name=self.l[self.n].split('_')
        name[2]='predict'
        name='_'.join(name)
        #self.predict_downsample=io.imread(self.path_predict_down+'\\'+name)
        self.merge=io.imread(self.path_merge+'\\'+name)
        self.mask=io.imread(self.path_mask+'\\'+name)
        self.random=io.imread(self.path_random+'\\'+name)
        self.smooth=io.imread(self.path_smooth+'\\'+name)
        self.predict1=io.imread(self.path_predict1+'\\'+name)
        self.predict2=io.imread(self.path_predict2+'\\'+name)
        self.predict3=io.imread(self.path_predict3+'\\'+name)
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()

    
        #self.ax1.imshow(img)
        self.ax1.imshow(self.label,cmap='gray')
        self.ax2.imshow(self.predict1,cmap='gray')
        self.ax3.imshow(self.predict2,cmap='gray')
        self.ax4.imshow(self.predict3,cmap='gray')
        self.ax5.imshow(self.merge,cmap='gray')

        

        self.fig.canvas.draw_idle()
        

        

    
        
def statistic():
    d=os.listdir(config.path_labels)
    n=numpy.array([0,0,0,0])
    for i in d:
        if 'spring' in i:
            n[0]=n[0]+1
        if 'summer' in i:
            n[1]=n[1]+1
        if 'fall' in i:
            n[2]=n[2]+1
        if 'winter' in i:
            n[3]=n[3]+1
    
    print(n)
    n=n/len(d)
    print(n)        

if __name__=='__main__':
    test=UI(mode='normal',init=0)
    #statistic()
    