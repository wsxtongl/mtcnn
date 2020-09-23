import mynet
import train

if __name__ == '__main__':
    net = mynet.ONet()

    trainer = train.Trainer(net, './param/onet.pt', r"D:\mtcnn_sample\48") # 网络，保存参数，训练数据；创建训器
    trainer.train()                                                     # 调用训练器中的train方法