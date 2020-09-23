import mynet
import train

if __name__ == '__main__':
    net = mynet.PNet()

    trainer = train.Trainer(net, './param/pnet.pt', r"D:\mtcnn_sample\12") # 网络，保存参数，训练数据；创建训器
    trainer.train()                                                     # 调用训练器中的train方法