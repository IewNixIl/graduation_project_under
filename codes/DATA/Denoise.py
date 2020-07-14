from skimage import io, filters
from skimage.morphology import disk
from matplotlib import pyplot
import time

if __name__=='__main__':
    path='G:\\M12\\dfc\\m1474000\\ROIs1868_summer\\s1_4'
    name='ROIs1868_summer_s1_4_p96.tif'
    n_r=2
    n_c=2
    n_filter=5
    img=io.imread(path+'\\'+name)

    img=(img+22)/22
    
    t1=time.time()
    r1=filters.median(img[:,:,0],disk(5))
    r2=filters.median(img[:,:,1],disk(5))
    t2=time.time()
    print(t2-t1)
    

    
    pyplot.subplot(n_r,n_c,1)
    pyplot.imshow(img[:,:,0],cmap='gray')
    pyplot.subplot(n_r,n_c,2)
    pyplot.imshow(img[:,:,1],cmap='gray')
    
    pyplot.subplot(n_r,n_c,3)
    pyplot.imshow(r1,cmap='gray')
    pyplot.subplot(n_r,n_c,4)
    pyplot.imshow(r2,cmap='gray')
    
    
    
    
    pyplot.show()
    