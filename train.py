import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from mydataset import Dataset # 导入数据集

DEVICE = "cuda:0"
# 创建训练器
class Trainer:
    def __init__(self, net, save_path,dataset_path): # 网络，参数保存路径，训练数据路径，cuda加速为True
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        #self.isCuda = isCuda

        # if self.isCuda:      # 默认后面有个else
        #     self.net.cuda() # 给网络加速
        self.net.to(DEVICE)
        # 创建损失函数
        # 置信度损失
        self.cls_loss_fn = nn.BCELoss() # ★二分类交叉熵损失函数，是多分类交叉熵（CrossEntropyLoss）的一个特例；用BCELoss前面必须用sigmoid激活,用CrossEntropyLoss前面必须用softmax函数
        # 偏移量损失
        self.offset_loss_fn = nn.MSELoss()

        # 创建优化器
        self.optimizer = optim.Adam(self.net.parameters())

        # 恢复网络训练---加载模型参数，继续训练
        if os.path.exists(self.save_path): # 如果文件存在，接着继续训练
            self.net.load_state_dict(torch.load(self.save_path))

    # 训练方法
    def train(self):
        faceDataset = Dataset(self.dataset_path) # 数据集
        dataloader = DataLoader(faceDataset, batch_size=512, shuffle=True,drop_last=True) # 数据加载器
        #num_workers=4：有4个线程在加载数据(加载数据需要时间，以防空置)；drop_last：为True时表示，防止批次不足报错。

        while True:
            for i, (img_data_, category_, offset_) in enumerate(dataloader): # 样本，置信度，偏移量
                #if self.isCuda:                    # cuda把数据读到显存里去了(先经过内存)；没有cuda在内存，有cuda在显存
                img_data_ = img_data_.to(DEVICE)  # [512, 3, 12, 12]
                category_ = category_.to(DEVICE) # 512, 1]
                offset_ = offset_.to(DEVICE)    # [512, 4]

                # 网络输出
                _output_category, _output_offset = self.net(img_data_) # 输出置信度，偏移量
                # print(_output_category.shape)     # [512, 1, 1, 1]
                # print(_output_offset.shape)       # [512, 4, 1, 1]
                output_category = _output_category.view(-1, 1) # [512,1]
                output_offset = _output_offset.view(-1, 4)     # [512,4]
                # output_landmark = _output_landmark.view(-1, 10)

                # 计算分类的损失----置信度
                category_mask = torch.lt(category_, 2)  # 对置信度小于2的正样本（1）和负样本（0）进行掩码; ★部分样本（2）不参与损失计算；符合条件的返回1，不符合条件的返回0
                category = torch.masked_select(category_, category_mask)              # 对“标签”中置信度小于2的选择掩码，返回符合条件的结果

                output_category = torch.masked_select(output_category, category_mask) # 预测的“标签”进掩码，返回符合条件的结果
                cls_loss = self.cls_loss_fn(output_category, category)                # 对置信度做损失

                # 计算bound回归的损失----偏移量
                offset_mask = torch.gt(category_, 0)  # 对置信度大于0的标签进行掩码；★负样本不参与计算,负样本没偏移量;[512,1]
                offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引；[244]
                offset = offset_[offset_index]                   # 标签里饿偏移量；[244,4]
                output_offset = output_offset[offset_index]      # 输出的偏移量；[244,4]
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 偏移量损失

                #总损失
                loss = cls_loss + offset_loss

                # 反向传播，优化网络
                self.optimizer.zero_grad() # 清空之前的梯度
                loss.backward()           # 计算梯度
                self.optimizer.step()    # 优化网络

                #输出损失：loss-->gpu-->cup（变量）-->tensor-->array
                print("i=",i ,"loss:", loss.cpu().detach().numpy(), " cls_loss:", cls_loss.cpu().detach().numpy(), " offset_loss",
                      offset_loss.cpu().detach().numpy())
                # 保存
                if i%1000 == 0:
                    torch.save(self.net.state_dict(), self.save_path)
                    #torch.save(self.net.state_dict(), './param/pnet{0}.pt'.format(i)) # state_dict保存网络参数，save_path参数保存路径
                    print("save success")                            # 每轮次保存一次；最好做一判断：损失下降时保存一次
