import torch.nn as nn
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os


save_path = "models/net.t"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2, 10),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        fc1_out = self.fc1(x)
        out = self.fc2(fc1_out)
        return out,fc1_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torchvision.datasets.MNIST(root="D:MNIST_data",download=True,train=True,transform=transforms.Compose([transforms.ToTensor()]))
train_loader = data.DataLoader(dataset=train_data,shuffle=True,batch_size=600)

def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    # plt.xlim(xmin=-100,xmax=100)
    # plt.ylim(ymin=-100,ymax=100)
    plt.title("epoch=%d" % epoch)
    plt.savefig('./minist_images/epoch=%d.jpg' % epoch)

    plt.pause(0.001)
    plt.show()
if __name__ == '__main__':
    net = Net().to(device)
    if os.path.exists(save_path):
        net = torch.load(save_path)
    loss_fun = nn.MSELoss().to(device)
    optimzer = torch.optim.Adam(net.parameters())


    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x = x.view(-1,784).to(device)
            target = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1).to(device)
            # target = y.to(device)
            out_put,feat = net(x)
            loss = loss_fun(out_put,target)

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            feat_loader.append(feat)
            label_loader.append((y))


            if i % 10 == 0:
                print(loss.item())
        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)

        visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        epoch+=1
        torch.save(net, save_path)

