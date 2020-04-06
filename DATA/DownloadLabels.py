'''下载像素级分类标签图片'''
import urllib.request
import time

def getUrls(page):
    '''page 下载的网页文件路径（.html)
    返回图片的链接列表
    '''
    l=[]
    with open(page) as f:
        s=f.readline()
        while s:
            t=[]
            for i in range(len(s)):
                if s[i]=='\"':
                    t.append(i)
                if len(t)==2:
                    str=s[t[0]+1:t[1]]
                    if '.tif' not in str:
                        t=[]
                        continue
                    l.append(str)
                    t=[]
            
            s=f.readline()
    
    return l
    
def download_img(img_url,path_save):
    request = urllib.request.Request(img_url)
    
    response = urllib.request.urlopen(request)
    
    img_name = img_url.split('/')[-1]
    filename = path_save+ '\\'+img_name
    if (response.getcode() == 200):
        with open(filename, "wb") as f:
            f.write(response.read()) # 将内容写入图片
        return filename

    
if __name__=='__main__':
    page='D:\\Study\\dataset\\FROMGLC10-2017\\FROM-GLC10-website.html'
    path_save='E:\\FROMGLC10-2017\\labels'
    result=getUrls(page)
    n=0
    all=len(result)
    for i in result:
        if n<884:
            n+=1
            continue
        download_img(i,path_save)
        n+=1
        print(str(n)+'/'+str(all))
        
    