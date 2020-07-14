'''关于标签（FROM-GLC10）的操作
投影坐标 转 经纬度时 ， 从经纬度转投影，依据的是使用投影坐标的图片；从投影转经纬，依据的也是投影坐标的图片
投影坐标和经纬度刚好旋转了180'''
from osgeo import gdal
from osgeo import osr
import numpy

CLASSNAME={10:'Cropland',
            20:'Forest',
            30:'Grassland',
            40:'Shrubland',
            50:'Wetland',
            60:'Water',
            70:'Tundra',
            80:'Impervious surface',
            90:'Bareland',
            100:'Snow/Ice'}

CLASSCOLOR={10:'#c24f44',
            20:'#009900',
            30:'#b6ff05',
            40:'#c6b044',
            50:'#27ff87',
            60:'#1c0dff',
            70:'#a5a5a5',
            80:'#fbff13',
            90:'#f9ffa4',
            100:'#69fff8'}
            
def getSRSPair(dataset):
    '''获得给定数据的投影参考坐标系、地理参考系
    '''
    prosrs=osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs=prosrs.CloneGeogCS()
    return prosrs,geosrs
    
def geo2lonlat(dataset,x,y):
    '''投影坐标转经纬度坐标
    dataset 的坐标系与x y相同
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs,geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]
    
def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]
    
def transform(dataset,path_labels):
    width=dataset.RasterXSize#X
    height=dataset.RasterYSize#Y
    geo=dataset.GetGeoTransform()
    
    left_up=geo2lonlat(dataset,geo[0],geo[3])
    
    print(left_up)
    #在from-glc10中找相应的图片
    #图片名经纬度是 左下角的  先纬度再精度
    lat=int(left_up[1]-left_up[1]%2)
    lon=int(left_up[0]-left_up[0]%2)
    print(lon)
    
    #图片名
    name='fromglc10v01_'+str(lat)+'_'+str(lon)+'.tif'
    print(name)
    
    #读图片
    dataset=gdal.Open(path_labels+'\\'+name)
    width=dataset.RasterXSize#X
    height=dataset.RasterYSize#Y
    band=dataset.RasterCount#波段数
    data=dataset.ReadAsArray(0,0,width,height)
    
    geo=dataset.GetGeoTransform()
    
    #找行列数
    left_up_r=(geo[3]-lat)/geo[5]
    left_up_c=(lon-geo[0])/geo[1]
    
    return data[left_up_r:left_up_r-256,left_up_c:left_up_c-256]
    
    

    
    
    
'''
def color(value):
    #将十六进制颜色代码转为三元数组
    digit = list(map(str, range(10))) + list("abcdef")
    a1 = digit.index(value[1]) * 16 + digit.index(value[2])
    a2 = digit.index(value[3]) * 16 + digit.index(value[4])
    a3 = digit.index(value[5]) * 16 + digit.index(value[6])
    return (((a1,a2,a3)))

def draw(img_gray):
    #将类别编码的灰度图转为三通道，便于显示
    r=img_gray.shape[0]
    c=img_gray.shape[1]
    result=numpy.zeros((r,c,3))
    
    for i in CLASSCOLOR:
        result=numpy.where(img_gray==i,numpy.array(color(CLASSCOLOR[i])),(((0,0,0))))
    return result
'''    
    
            

if __name__=='__main__':
    from matplotlib import pyplot
    
    path1='E:\\dfc\\m1474000\\ROIs1158_spring\\s1_100\\ROIs1158_spring_s1_100_p29.tif'
    dataset=gdal.Open(path1)
    #transform(dataset)
    #geo=dataset.GetGeoTransform()
    transform(dataset,'E:\\FROMGLC10-2017\\labels')
    
    '''
    left_up=[417903.59640957054,7572426.855528089]
    right_bottom=[420463.59640957054,7569866.855528089]
    
    left_up=geo2lonlat(dataset,left_up[0],left_up[1])
    right_bottom=geo2lonlat(dataset,right_bottom[0],right_bottom[1])
    
    print(left_up)
    print(right_bottom)
    
    #path1='D:\\Study\\dataset_contest\\s1_4\\s1_4\\ROIs1868_summer_s1_4_p30.tif'
    path2='E:\\FROMGLC10-2017\\labels\\fromglc10v01_0_8.tif'
    #path1='E:\\FROMGLC10-2017\\labels\\fromglc10v01_0_10.tif'
    
    dataset1=gdal.Open(path2)
    width1=dataset1.RasterXSize#X
    height1=dataset1.RasterYSize#Y
    band1=dataset1.RasterCount#波段数
    
    geo1=dataset1.GetGeoTransform()
    x=geo1[0]+width1*geo1[1]+height1*geo1[2]
    y=geo1[3]+width1*geo1[4]+height1*geo1[5]
    
    print(geo1)
    print(x)
    print(y)
    '''
    
    
    
    
    
    
    