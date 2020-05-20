import cartopy.crs as ccrs
from matplotlib import pyplot
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import gdal
import osr
import pyproj
import warnings
import shelve
import numpy

warnings.filterwarnings('ignore')

def getcor(path):
    '''给定图片的路径
    返回坐标
    '''
    d = gdal.Open(path)
    proj = osr.SpatialReference(wkt=d.GetProjection())
    space = proj.GetAttrValue('AUTHORITY',1)
    
    adfGeoTransform = d.GetGeoTransform()
    
    # 左上角地理坐标
    x=adfGeoTransform[0]
    y=adfGeoTransform[3]
    
    
    p1 = pyproj.Proj(init="epsg:"+str(space)) # 定义数据地理坐标系 WGS84
    p2 = pyproj.Proj(init="epsg:4326") # 定义转换投影坐标系
    x2, y2 = pyproj.transform(p1,p2,x,y) # lon 和lat 可以是元组
    return x2,y2
    
def getname(path,namelist):
    '''给定namelist
    返回路径
    '''
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
    
def transform(name):
    '''与 getname 功能相反
    '''
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
    
def namelist_2_corlist(namelists):
    result=[]
    for i in range(len(namelists)):
        path=getname('G:\\dataset\\SEN12MS\\m1474000',namelists[i])
        x,y=getcor(path)
        result.append([x,y])
        if i%100==0:
            print(i)
        
    return result
        
    
'''
with shelve.open('D:\\Codes\\Graduation\\Graduation\\train_test_list') as f:
    tr=f['test']
r=namelist_2_corlist(tr)
with shelve.open('D:\\Codes\\Graduation\\Graduation\\DATA\\xy') as f:
    f['test']=r

'''

with shelve.open('D:\\Codes\\Graduation\\Graduation\\DATA\\xy') as f:
    tr=f['train']
    te=f['test']
tr=numpy.array(tr)
te=numpy.array(te)
x_r=tr[:,0]
y_r=tr[:,1]
x_e=te[:,0]
y_e=te[:,1]

    



pyplot.figure(figsize=(8, 10))
ax = pyplot.axes(projection=ccrs.PlateCarree(central_longitude=0))
ax.stock_img()
ax.scatter(x_r,y_r,marker='v',s=6,c='k')
ax.scatter(x_e,y_e,marker='v',s=6,c='r')
# 标注坐标轴
ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
pyplot.show()

    




'''
pyplot.figure(figsize=(8, 10))
ax = pyplot.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.stock_img()
ax.scatter(x,y)
# 标注坐标轴
ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
pyplot.show()


path = 'G:\\dataset\\SEN12MS\\m1474000\\ROIs1158_spring\\lc_40\\ROIs1158_spring_lc_40_p30.tif'
d = gdal.Open(path)
proj = osr.SpatialReference(wkt=d.GetProjection())
space = proj.GetAttrValue('AUTHORITY',1)
print(space)

adfGeoTransform = d.GetGeoTransform()

# 左上角地理坐标
x=adfGeoTransform[0]
y=adfGeoTransform[3]


p1 = pyproj.Proj(init="epsg:32724") # 定义数据地理坐标系 WGS84
p2 = pyproj.Proj(init="epsg:4326") # 定义转换投影坐标系
x2, y2 = pyproj.transform(p1,p2,x,y) # lon 和lat 可以是元组
print(x2,y2)


pyplot.figure(figsize=(8, 10))
ax = pyplot.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.stock_img()
# 标注坐标轴
ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
pyplot.show()
'''

