# MTCNN的使用
# 流程：图像-->缩放-->P网(NMS和边界框回归)-->R网路(NMS和边界框回归)--->O网络(NMS和边界框回归)

import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from tool import utils
import nets
from torchvision import transforms
import time
import os
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from torchvision import models
from sql import encoding_FaceStr,Connect
# 网络调参
# P网络:
p_cls = 0.6 #原为0.6
p_nms = 0.5 #原为0.5
# R网络：
r_cls = 0.6 #原为0.6
r_nms = 0.5 #原为0.5
#
# O网络：
o_cls = 0.9 #原为0.97
o_nms = 0.7 #原为0.7

device = "cuda" if torch.cuda.is_available() else "cpu"
# 侦测器
class Detector:
    # 初始化时加载三个网络的权重(训练好的)，cuda默认设为True
    def __init__(self, pnet_param="./param_0/pnet.pt", rnet_param="./param_0/rnet.pt",onet_param="./param_0/onet.pt",
                 isCuda=True):

        self.isCuda = isCuda

        self.pnet = nets.PNet() # 创建实例变量，实例化P网络
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda() # 给P网络加速
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param)) # 把训练好的权重加载到P网络中
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        # self.pnet.eval() # 训练网络里有BN（批归一化时），要调用eval方法，使用是不用BN，dropout方法
        # self.rnet.eval()
        # self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ]) # 图片数据类型转换

    def detect(self, image): # 检测图片

        # P网络检测-----1st
        start_time = time.time()               # 开始计时
        pnet_boxes = self.__pnet_detect(image) # 调用__pnet_detect函数（后面定义）

        if pnet_boxes.shape[0] == 0:           # 若P网络没有人脸时，避免数据出错，返回一个新数组
            return np.array([])
        end_time = time.time()                 # 计时结束
        t_pnet = end_time - start_time         # P网络所占用的时间差
        #print("t_pnet:"f"{t_pnet}")
        #return pnet_boxes                    # p网络检测出的框

        #R网络检测-------2nd
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes) # 传入原图，P网络的一些框，根据这些框在原图上抠图
        #print(rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        # #O网络检测--------3rd
        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes) # 把原图和R网络里的框传到O网络里去

        if onet_boxes.shape[0] == 0:                      # 若P网络没有人脸时，避免数据出错，返回一个新数组
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        # 三网络检测的总时间
        t_sum = t_pnet + t_rnet + t_onet
        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    # 创建P网检测函数
    def __pnet_detect(self, image): # ★p网络全部是卷积，与输入图片大小无关，可输出任意形状图片

        boxes = [] # 创建空列表,接收符合条件的建议框
        box1 = np.random.randn(1,5)
        img = image
        w, h = img.size
        min_side_len = min(w, h) # 获取图片的最小边长

        scale = 1 # 初始缩放比例（为1时不缩放）:得到不同分辨率的图片
        start = time.time()
        while min_side_len > 12: # 直到缩放到小于等于12时停止
            img_data = self.__image_transform(img) # 将图片数组转成张量

            img_data = img_data.cuda() # 将图片tensor传到cuda里加速
            img_data.unsqueeze_(0) # 在“批次”上升维（测试时传的不止一张图片）
            # print("img_data:",img_data.shape) # [1, 3, 416, 500]：C=3,W=416,H=500

            _cls, _offest = self.pnet(img_data) # ★★返回多个置信度和偏移量
            # print("_cls",_cls.shape)         # [1, 1, 203, 245]:NCWH：分组卷积的特征图的通道和尺寸★
            # print("_offest", _offest.shape) # [1, 4, 203, 245]:NCWH

            cls= _cls[0][0].cpu().data # [203, 245]：分组卷积特征图的尺寸：W,H
            offest = _offest[0].cpu().data #[4, 203, 245] # 分组卷积特征图的通道、尺寸:C,W,H

            idxs = torch.nonzero(torch.gt(cls, p_cls))# ★置信度大于0.6的框索引；把P网络输出，看有没没框到的人脸，若没框到人脸，说明网络没训练好；或者置信度给高了、调低

            box2 = self.__boox(idxs, offest, cls[idxs[:,0], idxs[:,1]], scale)
            box1 =np.vstack((box1, box2))

            # for idx in idxs: # 根据索引，依次添加符合条件的框；cls[idx[0], idx[1]]在置信度中取值：idx[0]行索引，idx[1]列索引
            #     boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale)) # ★调用框反算函数_box（把特征图上的框，反算到原图上去），把大于0.6的框留下来；

            scale *= 0.7 # 缩放图片:循环控制条件
            _w = int(w * scale) # 新的宽度
            _h = int(h * scale)

            img = img.resize((_w, _h)) # 根据缩放后的宽和高，对图片进行缩放
            min_side_len = min(_w, _h) # 重新获取最小宽高

        # return np.array(boxes)
        #return utils.nms(np.array(boxes), p_nms) #返回框框，原阈值给p_nms=0.5（iou为0.5），尽可能保留IOU小于0.5的一些框下来，若网络训练的好，值可以给低些


        return utils.nms(box1[1:], p_nms)
    # 特征反算：将回归量还原到原图上去，根据特征图反算的到原图建议框
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12): # p网络池化步长为2
        #通过特征图，反算原图的候选区域（建议框）
        _x1 = (start_index[1].float() * stride) / scale # 索引乘以步长，除以缩放比例；★特征反算时“行索引，索引互换”，原为[0]
        _y1 = (start_index[0].float() * stride) / scale
        _x2 = (start_index[1].float() * stride + side_len) / scale-1
        _y2 = (start_index[0].float() * stride + side_len) / scale-1

        # ow = _x2 - _x1  # 人脸所在区域建议框的宽和高
        # oh = _y2 - _y1
        ow = side_len/scale
        oh = side_len/scale
        #通过候选区域，反算网络相对于原图的实际框
        _offset = offset[:, start_index[0], start_index[1]] # 根据idxs行索引与列索引，找到对应偏移量△δ:[x1,y1,x2,y2]
        x1 = _x1 + ow * _offset[0] # 根据偏移量算实际框的位置，x1=x1_+w*△δ；生样时为:△δ=x1-x1_/w
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]  # 正式框：返回4个坐标点和1个偏移量

    def __boox(self,start_index, offset, cls, scale, stride=2, side_len=12):  # p网络池化步长为2
        # 通过特征图，反算原图的候选区域（建议框）

        _x1 = (start_index[:,1].float() * stride) / scale  # 索引乘以步长，除以缩放比例；★特征反算时“行索引，索引互换”，原为[0]
        _y1 = (start_index[:,0].float() * stride) / scale
        _x2 = (start_index[:,1].float() * stride + side_len) / scale -1
        _y2 = (start_index[:,0].float() * stride + side_len) / scale -1

        # ow1 = _x2 - _x1  # 人脸所在区域建议框的宽和高
        # oh1 = _y2 - _y1
        ow = side_len / scale
        oh = side_len / scale

        # 通过候选区域，反算网络相对于原图的实际框
        _offset = offset[:, start_index[:,0], start_index[:,1]]  # 根据idxs行索引与列索引，找到对应偏移量△δ:[x1,y1,x2,y2]
        x1 = _x1 + ow * _offset[0]  # 根据偏移量算实际框的位置，x1=x1_+w*△δ；生样时为:△δ=x1-x1_/w
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]
        x1 = x1.reshape(-1, 1)
        y1 = y1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        y2 = y2.reshape(-1, 1)
        cls = cls.reshape(-1, 1).float()
        cat = torch.cat((x1, y1, x2, y2, cls), 1)


        return cat.numpy()


    # 创建R网络检测函数
    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = [] # 创建空列表，存放抠图
        _pnet_boxes = utils.convert_to_square(pnet_boxes) # ★给p网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”，再抠图

        for _box in _pnet_boxes: # ★遍历每个框，每个框返回框4个坐标点，抠图，放缩，数据类型转换，添加列表
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])


            img = image.crop((_x1, _y1, _x2, _y2)) # 根据4个坐标点抠图
            img = img.resize((24, 24)) # 放缩在固尺寸
            img_data = self.__image_transform(img) # 将图片数组转成张量
            _img_dataset.append(img_data)


        img_dataset =torch.stack(_img_dataset) # stack堆叠(默认在0轴)，此处相当数据类型转换，见例子2★
        if self.isCuda:
            img_dataset = img_dataset.cuda() # 给图片数据采用cuda加速

        _cls, _offset = self.rnet(img_dataset) # ★★将24*24的图片传入网络再进行一次筛选

        cls = _cls.cpu().data.numpy() # 将gpu上的数据放到cpu上去，在转成numpy数组

        offset = _offset.cpu().data.numpy()
        # print("r_cls:",cls.shape)  # (11, 1):P网络生成了11个框
        # print("r_offset:", offset.shape)  # (11, 4)

        boxes = [] #R 网络要留下来的框，存到boxes里

        idxs, _ = np.where(cls > r_cls) # 原置信度0.6是偏低的，时候很多框并没有用(可打印出来观察)，可以适当调高些；idxs置信度框大于0.6的索引；★返回idxs:0轴上索引[0,1]，_:1轴上索引[0,0]，共同决定元素位置，见例子3
        _box = _pnet_boxes[np.array(idxs)]

        for idx in idxs: # 根据索引，遍历符合条件的框；1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1 # 基准框的宽
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0] # 实际框的坐标点
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]]) # 返回4个坐标点和置信度




        return utils.nms(np.array(boxes), r_nms) # 原r_nms为0.5（0.5要往小调），上面的0.6要往大调;小于0.5的框被保留下来


    # 创建O网络检测函数
    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = [] # 创建列表，存放抠图r
        _rnet_boxes = utils.convert_to_square(rnet_boxes) # 给r网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”
        for _box in _rnet_boxes: # 遍历R网络筛选出来的框，计算坐标，抠图，缩放，数据类型转换，添加列表，堆叠
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2)) # 根据坐标点“抠图”
            img = img.resize((48, 48))
            img_data = self.__image_transform(img) # 将抠出的图转成张量
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset) # 堆叠，此处相当数据格式转换，见例子2
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)
        cls = _cls.cpu().data.numpy()       # (1, 1)
        offset = _offset.cpu().data.numpy() # (1, 4)

        boxes = [] # 存放o网络的计算结果
        idxs, _ = np.where(cls > o_cls) # 原o_cls为0.97是偏低的，最后要达到标准置信度要达到0.99999，这里可以写成0.99998，这样的话出来就全是人脸;留下置信度大于0.97的框；★返回idxs:0轴上索引[0]，_:1轴上索引[0]，共同决定元素位置，见例子3
        for idx in idxs: # 根据索引，遍历符合条件的框；1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）
            _box = _rnet_boxes[idx] # 以R网络做为基准框
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1 # 框的基准宽，框是“方”的，ow=oh
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0] # O网络最终生成的框的坐标；生样，偏移量△δ=x1-_x1/w*side_len
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]]) #返回4个坐标点和1个置信度

        return utils.nms(np.array(boxes), o_nms, isMin=True) # 用最小面积的IOU；原o_nms(IOU)为小于0.7的框被保留下来

def Test_image():
    path = r"test_images"
    i = 0
    for file in os.listdir(path):

        # cap = cv2.VideoCapture(r"C:\Users\Administrator\(练习室版) (G)I-DLE - LATATA[00].mp4")

        # while (cap.isOpened()):
        #     ret, frame = cap.read()
        #
        #     img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # img = Image.open(os.path.join(path, file))
        img = cv2.imread(os.path.join(path, file))
        print(img.shape)
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)

        print("size:", im.size)
        detector = Detector()
        boxes = detector.detect(im)

        boxes = utils.square(boxes)
        # imDraw = ImageDraw.Draw(img)
        for box in boxes:  # 多个框，没循环一次框一个人脸
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            print((x1, y1, x2, y2, box[4]))
            face_crop = im.crop((x1, y1, x2, y2))  # 抠图
            face_resize = face_crop.resize((145, 145), Image.ANTIALIAS)
            #face_resize.save("./needface_image/{0}.jpg".format(i))
            i+=1
            # print("conf:",box[4]) # 置信度
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            # im.show() # 每循环一次框一个人脸
        cv2.namedWindow("frame", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
        cv2.resizeWindow("frame", 800, 500)  # 设置长和宽
        cv2.imshow("frame", img)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()

def Crop_Video():
    cap = cv2.VideoCapture(0)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if i%10==0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img)

            print("size:", im.size)
            detector = Detector()
            boxes = detector.detect(im)

            boxes = utils.square(boxes)
                # imDraw = ImageDraw.Draw(img)
            for box in boxes:  # 多个框，没循环一次框一个人脸
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                print((x1, y1, x2, y2, box[4]))
                face_crop = im.crop((x1, y1, x2, y2))  # 抠图
                face_resize = face_crop.resize((145, 145), Image.ANTIALIAS)
                face_resize.save("./needface_image/{0}.jpg".format(i))
                #i += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        i += 1
        cv2.namedWindow("frame", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
        cv2.resizeWindow("frame", 544, 960)  # 设置长和宽
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

def Show_Video():
    cap = cv2.VideoCapture(0)

    nam = " "
    connect = Connect()
    tuple_data = connect.load_faceData()

    while (cap.isOpened()):


            ret, frame = cap.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(img)

            print("size:", im.size)

            detector = Detector()
            boxes = detector.detect(im)


            boxes = utils.square(boxes)
                    # imDraw = ImageDraw.Draw(img)
            for box in boxes:  # 多个框，没循环一次框一个人脸
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                print((x1, y1, x2, y2, box[4]))
                if x1 <0 or y1<0 or x2<0 or y2<0:
                    continue
                l = y2-y1
                image = img[y1:y2,x1:x1+l]

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (100, 100))
                tran = transform
                im_face = tran(image).to(device)
                im_face = torch.unsqueeze(im_face, 0).to(device)
                model.eval()
                feat = model(im_face)
                fe = feat.cpu().detach()
                fe = torch.squeeze(fe)

                name = tuple_data[0]
                face_data = tuple_data[1]

                name_list = []
                cos_list = []
                for i in range(len(name)):
                    n = name[i]

                    data_t = torch.Tensor(face_data[i])

                    cos_torch = compare(fe, data_t)
                    print(n, cos_torch)
                    cos_data = cos_torch[cos_torch >0.8]
                    if len(cos_data) != 0:
                        cos = torch.max(cos_data)
                        cos_list.append(cos)
                        name_list.append(n)
                if len(cos_list) > 0:
                    index = cos_list.index(max(cos_list))
                    nam = name_list[index]
                    print(nam, cos_list[index])

                # print(len(nam_list))
                cv2.namedWindow("frame", 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细，字体类型
                cv2.resizeWindow("frame", 640, 480)
                imgg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #print(len(cos_list))
                if len(boxes) == 0 or len(cos_list) == 0:
                    nam = "miss"
                    cv2.rectangle(imgg, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    frame = cv2.putText(imgg, nam, (int(x1+l/3), int(y1-l/10)), font, 0.5, (0, 0, 255), 1, )
                else:
                    cv2.rectangle(imgg, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    frame = cv2.putText(imgg, nam, (int(x1+l/3), int(y1-l/10)), font, 0.5, (0, 0, 255), 1, )
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
    cv2.destroyAllWindows()

def Crop_image():
    image_path = r"./face_imagev5"
    path = r"./detect_imgv5"
    path1 = os.listdir(path)
    path2 = os.listdir(image_path)
    path1.sort(key=lambda x: int(x[:]))
    path2.sort(key=lambda x: int(x[:]))
    print(path1)
    print(path2)
    # plt.figure()
    # plt.ion()

    for i in range(1):
        file_name = path2[i]
        j = 0
        for file_img in os.listdir(os.path.join(image_path,file_name)):
            print(file_img)
            detector = Detector()

            with Image.open(os.path.join(os.path.join(image_path, file_name),file_img)) as im:  # 打开图片

                boxes = detector.detect(im)
                print(boxes)

                boxes = utils.square(boxes)
                #imDraw = ImageDraw.Draw(im)
                for box in boxes:  # 多个框，没循环一次框一个人脸
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    print((x1, y1, x2, y2, box[4], file_img))
                    face_crop = im.crop((x1, y1, x2, y2))  # 抠图
                    face_resize = face_crop.resize((145, 145), Image.ANTIALIAS)
                    face_resize.save("./detect_imgv5/{0}/{1}.jpg".format(path1[i], j))
                    j += 1


def compare(face1,face2):
    face1_norm = nn.functional.normalize(face1,dim=0)
    face2_norm = nn.functional.normalize(face2,dim=1)
    casa = torch.matmul(face1_norm,face2_norm.t())
    return casa
if __name__ == '__main__':
    path1 = "./face_model/model.120t"
    path2 = "./face_model/arc.t"
    model = models.densenet121(pretrained=True)
    model = model.to(device)
    if os.path.exists(path1):
        model.load_state_dict(torch.load(path1))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ])
    video = Show_Video()
    #test = Test_image()
    #crop = Crop_Video()








# 备注：以上提到的例子1、2、3见“notes/13-detect”
