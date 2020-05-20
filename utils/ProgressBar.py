import time

class Bar:
    def __init__(self,name,max_batch,max_epoch,fill='▮',length=40):
        '''name 显示在进度条前的文字
        max_batch 最大batch数
        max_epoch 最大epoch数
        fill 进度条填充字符
        length 实际画进度条时，画多少字符，默认画50个
        '''
        self.name=name
        self.max_batch=max_batch
        self.max_epoch=max_epoch
        self.fill=fill
        self.length=length
        self.number=0
        self.number_epoch=1
        self.draw()

    def draw(self,info=None):
        '''info 追加的信息 一个字典 比如{loss:0.3,iou:0.4} 则在最后打印 [loss:0.3] [iou:0.4]
        '''
        self.info=info
        namelist=[self.number,self.max_batch,self.number_epoch,self.max_epoch]
        if self.info:
            for i in self.info:
                namelist.append(self.info[i])
        namelist=tuple(namelist)
        r=self.writeLine()
        if self.number>=self.max_batch:
            print(r%namelist)
            if self.number_epoch<self.max_epoch:
                self.number=0
                self.number_epoch=self.number_epoch+1
                namelist=[self.number,self.max_batch,self.number_epoch,self.max_epoch]
                if self.info:
                    for i in self.info:
                        namelist.append(self.info[i])
                namelist=tuple(namelist)
                r=self.writeLine()
                print(r%namelist,end='')
                self.number=self.number+1
            else:
                return
        else:
            print(r%namelist,end='')
            self.number=self.number+1

    def writeLine(self):
        r='\r'+self.name+' |'
        now=int(self.number/self.max_batch*self.length)
        for i in range(self.length):
            if i+1<=now:
                r=r+self.fill
            else:
                r=r+' '
        r=r+'| '
        r=r+'[%d/%d] '
        #r=r+'['+str(round(self.number/self.max*100,1))+'%] '
        r=r+'[%d/%d] '
        if self.info:
            for i in self.info:
                r=r+'['+i+':%.5f] '
        return r



if __name__=='__main__':
    from TorchBoard import Board
    name='test'
    max_batch=40
    max_epoch=3
    fill='▮'
    bar=Bar(name,max_batch,max_epoch)
    board=Board(False)
    loss=[]
    iou=[]

    l=10
    io=0
    for i in range(max_epoch):
        for j in range(max_batch):
            l-=0.1
            io+=0.01
            board.setData([l,io])
            loss.append(l)
            iou.append(io)

            time.sleep(0.2)
            info={'loss:':loss[-1],'iou:':iou[-1]}
            bar.draw(info=info)
    board.closeClient()
