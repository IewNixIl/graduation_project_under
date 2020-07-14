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
    segment=slic(img,n_segments=n_segments, compactness=compactness,enforce_connectivity=True)
    out=mark_boundaries(img,segment,color=[0,0,0.2])
    
    #img=exposure.adjust_gamma(img,0.5)
    #out=exposure.adjust_gamma(out,0.5)
    
    wdi=(i[:,:,3]-i[:,:,1])/(i[:,:,3]+i[:,:,1])
    
    wdi=(wdi/wdi.max())*255
    
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
        self.path_label=config.path_labels
        if self.mode=='normal':
            with shelve.open(config.path_devision) as f:
                self.imglist=f['test']
        else:
            self.imglist=os.listdir(config.path_labels)

        self.n=init
        
        
        self.ifpress=False
        self.ifloadlabel=False
        
        fig=pyplot.figure()
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.mpl_connect('key_press_event',self.on_key_press)
        fig.canvas.mpl_connect('button_press_event',self.on_button_press)
        fig.canvas.mpl_connect('motion_notify_event',self.on_button_move)
        fig.canvas.mpl_connect('button_release_event',self.on_button_release)
        
        self.fig=fig
        self.ax1=fig.add_subplot(3,2,1)
        self.ax2=fig.add_subplot(3,2,3)
        self.ax4=fig.add_subplot(3,2,5)
        self.ax3=fig.add_subplot(1,2,2)
        pyplot.get_current_fig_manager().window.state('zoomed')
        #self.ax2=fig.add_subplot(1,2,2)
        
        
        
        self.valuelist=[]
        self.label=numpy.zeros((256,256))
        self.ifloadlabel=True
        self.draw()
        
        
        
        pyplot.show()
        
    def on_key_press(self,event):
        if event.key=='a' or event.key=='left':
            self.n-=1
            print(self.n)
            self.valuelist=[]
            self.label=numpy.zeros(self.segment.shape)
            self.ifloadlabel=True
            self.draw()

        if event.key=='d' or event.key=='right':
            if self.n+1>=len(self.imglist):
                return
            self.n+=1
            print(self.n)
            self.valuelist=[]
            self.label=numpy.zeros(self.segment.shape)
            self.ifloadlabel=True
            self.draw()

        if event.key=='e' or event.key=='enter':
            self.save_label()
            
        if event.key=='Q':
            f=numpy.unique(self.segment).tolist()
            for i in f:
                if i not in self.valuelist:
                    self.valuelist.append(i)
            for i in range(len(self.valuelist)):
                if i==0:
                    flag=(self.segment==self.valuelist[i])
                else:
                    flag=flag+(self.segment==self.valuelist[i])
            self.label=numpy.where(flag,1.0,0)
            
            self.draw()
            
        
    def on_button_press(self,event):
        
        try:
            r=int(event.ydata)
            c=int(event.xdata)
        except TypeError:
            return
        value=self.segment[r,c]
        if event.button==1:
            if value not in self.valuelist:
                self.ifpress=True
                self.valuelist.append(value)
        elif event.button==3:
            if value in self.valuelist:
                self.ifpress=True
                self.valuelist.remove(value)
            
            
    def on_button_move(self,event):
        if not self.ifpress:
            return
            
        try:
            r=int(event.ydata)
            c=int(event.xdata)
        except TypeError:
            return
        value=self.segment[r,c]
        if event.button==1:
            if value not in self.valuelist:
                self.valuelist.append(value)
        elif event.button==3:
            if value in self.valuelist:
                self.valuelist.remove(value)
            
    def on_button_release(self,event):
        if not self.ifpress:
            return
        self.ifpress=False
        for i in range(len(self.valuelist)):
            if i==0:
                flag=(self.segment==self.valuelist[i])
            else:
                flag=flag+(self.segment==self.valuelist[i])
        self.label=numpy.where(flag,1,0).astype(int)
        self.draw()
    
        
    def draw(self):
        
        if self.mode=='normal':
            segment,out,img,wdi=seg(getname(config.path,self.imglist[self.n]))
        else:
            
            segment,out,img,wdi=seg(getname(config.path,transform(self.imglist[self.n])))
        self.segment=segment
        if self.ifloadlabel:
            self.read_label()
            self.ifloadlabel=False
        #self.ax1.imshow(out)
        t=numpy.where(self.label==1,0.5,out[:,:,2])
        out[:,:,2]=t
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax1.imshow(img)
        self.ax2.imshow(wdi,cmap='gray')
        self.ax3.imshow(out)
        self.ax4.imshow(self.label,cmap='gray')
        
        d=os.listdir(config.path_labels)
        self.ax3.set_title(str(len(d))+'/'+str(self.n+1))
        self.fig.canvas.draw_idle()
        
    def save_label(self):
        
        
        
        label=self.label*255
        label=label.astype(numpy.uint8)
        label=Image.fromarray(label)
        if self.mode=='normal':
            name=getname(config.path,self.imglist[self.n]).split('\\')[-1]
            name=name.split('_')
            name[2]='label'
            name='_'.join(name)
        else:
            name=self.imglist[self.n]
        label.save(self.path_label+'\\'+name)
        
    def read_label(self):
        
        dirlist=os.listdir(self.path_label)
        if self.mode=='normal':
            name=getname(config.path,self.imglist[self.n]).split('\\')[-1]
            name=name.split('_')
            name[2]='label'
            name='_'.join(name)
        else:
            name=self.imglist[self.n]
        if name in dirlist:
            self.label=numpy.array(Image.open(self.path_label+'\\'+name))/255
            self.label=self.label.astype(int)
            self.valuelist=list(numpy.unique(numpy.where(self.label==1,self.segment,-2)))
            self.valuelist.remove(-2)
    
        
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
    test=UI(mode='normal',init=100)
    #statistic()
    