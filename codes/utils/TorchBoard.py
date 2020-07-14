import time
import numpy as np
from matplotlib import pyplot
import socket


class Board:
    def __init__(self,ifserver):
        self.HOST='localhost'
        self.PORT=50007
        self.s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        if not ifserver:
            self.client()
        else:
            pyplot.ion()
            self.end=False
            self.fig=pyplot.figure(figsize=(9,7))
            self.ax1=self.fig.add_subplot(1,2,1)
            self.ax2=self.fig.add_subplot(1,2,2)

            self.loss=[]
            self.acc=[]
            self.draw()
            print('开始监听')
            self.server()
            print('结束监听')
            pyplot.ioff()
            pyplot.show()

    def client(self):
        self.s.connect((self.HOST,self.PORT))

    def server(self):
        self.s.bind((self.HOST,self.PORT))
        self.s.listen(1)
        while not self.end:
            conn,addr=self.s.accept()
            while True:

                msg=conn.recv(1024).decode()
                if not msg:
                    break
                if msg=='end':
                    self.end=True
                    break

                msg=msg[1:-1].split(',')
                self.loss.append(float(msg[0]))
                self.acc.append(float(msg[1]))
                self.draw()



        self.s.close()

    def draw(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.set_xlim([-1,110])
        self.ax2.set_xlim([-1,110])
        self.ax1.set_ylim([0,0.4])
        self.ax2.set_ylim([0.2,1])
        self.ax1.plot(self.loss)
        self.ax2.plot(self.acc)
        pyplot.pause(.1)


    def setData(self,data):
        data=str(data).encode()
        self.s.send(data)

    def  closeClient(self):
        data='end'.encode()
        self.s.send(data)
        self.s.close()



if __name__=='__main__':
    ifserver=True
    board=Board(ifserver)
    if not ifserver:
        #board.setData([1,0.4])
        board.closeClient()
