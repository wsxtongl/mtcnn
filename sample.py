import os
import traceback
from PIL import Image
import numpy as np




def iou(box, boxes, isMin = False): #1st框，一堆框，inMin(IOU有两种：一个除以最小值，一个除以并集)
    #计算面积：[x1,y1,x2,y3]
    box_area = (box[2] - box[0]) * (box[3] - box[1]) #原始框的面积
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  #数组代替循环

    #找交集：
    xx1 = np.maximum(box[0], boxes[:, 0]) #横坐标，左上角最大值
    yy1 = np.maximum(box[1], boxes[:, 1]) #纵坐标，左上角最大值
    xx2 = np.minimum(box[2], boxes[:, 2]) #横坐标，右下角最小值
    yy2 = np.minimum(box[3], boxes[:, 3]) #纵坐标，右小角最小值

    # 判断是否有交集
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    #交集的面积
    inter = w * h  #对应位置元素相乘
    if isMin: #若果为Ture
        ovr = np.true_divide(inter, np.minimum(box_area, area)) #最小面积的IOU：O网络用
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))  #并集的IOU：P和R网络用；交集/并集

    return ovr
def nms(boxes, thresh=0.3, isMin = False):
    #框的长度为0时(防止程序有缺陷报错)
    if boxes.shape[0] == 0:
        return np.array([])

    #框的长度不为0时
    #根据置信度排序：[x1,y1,x2,y2,C]
    _boxes = boxes[(-boxes[:, 4]).argsort()] # #根据置信度“由大到小”，默认有小到大（加符号可反向排序）
    #创建空列表，存放保留剩余的框
    r_boxes = []
    # 用1st个框，与其余的框进行比较，当长度小于等于1时停止（比len(_boxes)-1次）
    while _boxes.shape[0] > 1: #shape[0]等价于shape(0),代表0轴上框的个数（维数）
        #取出第1个框
        a_box = _boxes[0]
        #取出剩余的框
        b_boxes = _boxes[1:]

        #将1st个框加入列表
        r_boxes.append(a_box) ##每循环一次往，添加一个框
        # print(iou(a_box, b_boxes))

        #比较IOU，将符合阈值条件的的框保留下来
        index = np.where(iou(a_box, b_boxes,isMin) < thresh) #将阈值小于0.3的建议框保留下来，返回保留框的索引
        _boxes = b_boxes[index] #循环控制条件；取出阈值小于0.3的建议框

    if _boxes.shape[0] > 0: ##最后一次，结果只用1st个符合或只有一个符合，若框的个数大于1；★此处_boxes调用的是whilex循环里的，此判断条件放在循环里和外都可以（只有在函数类外才可产生局部作用于）
        r_boxes.append(_boxes[0]) #将此框添加到列表中
    #stack组装为矩阵：:将列表中的数据在0轴上堆叠（行方向）
    return np.stack(r_boxes)
box_label = r"D:\BaiduNetdiskDownload\CelebA\Anno\list_bbox_celeba.txt"
img_path = r"D:\BaiduNetdiskDownload\CelebA\Img\img_celeba.7z\img_celeba"
crop_imgpath = r"D:\mtcnn_testimg\1"
save_path = r"D:\mtcnn_sample"
for face_size in [12,24,48]:
    positive_dir = os.path.join(save_path,str(face_size),"positive")
    negative_dir = os.path.join(save_path,str(face_size),"negative")
    part_dir = os.path.join(save_path,str(face_size),"part")
    for dir in [positive_dir,negative_dir,part_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    positive_label = os.path.join(save_path,str(face_size),"positive_txt")
    negative_label = os.path.join(save_path,str(face_size),"negative_txt")
    part_label = os.path.join(save_path,str(face_size),"part_txt")
    count = 0
    coun = 0
    cou = 0
    try:
        positive_file = open(positive_label,"w")
        negative_file = open(negative_label,"w")
        part_file = open(part_label,"w")
        for i, line in enumerate(open(box_label)):
            if i < 2:
                continue  # i小于2时继续读文件readlines
            try:
                # strs = line.strip().split(" ")  # strip删除两边的空格
                # strs = list(filter(bool, strs))  # 过滤序列，过滤掉不符合条件的元素
                strs = line.split()              #以空格分割
                image_filename = strs[0]         #图像名
                image_file_path =os.path.join(img_path,image_filename)    #绝对路径
                with Image.open(image_file_path) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1])
                    y1 = float(strs[2])
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = x1 + w
                    y2 = y1 + h  #建议框坐标
                    if max(w,h)<40:      #去除不符合条件的图像
                        continue
                    boxes = [[x1, y1, x2, y2]]

                    x0 = x1 + w/2        #中心点坐标
                    y0 = y1 + h/2
                    for i in range(5):
                        w_ = np.random.randint(-w * 0.2, w * 0.2)  # 框的横向偏移范围：向左、向右移动了20%
                        h_ = np.random.randint(-h * 0.2, h * 0.2)
                        w0 = w_+x0
                        h0 = h_+y0

                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(
                            1.25 * max(w, h)))  # 边长偏移的随机数的范围；ceil大于等于该值的最小整数（向上取整）;原0.8
                        x1_ = max(w0 - side_len / 2,0)  # 坐标点随机偏移
                        y1_ = max(h0- side_len / 2,0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len
                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len  # 偏移量△δ=(x1-x1_)/side_len;新框的宽度;★????还要梳理，可打印出来观察
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])
                        face_crop = img.crop(crop_box)  # “抠图”，crop剪下框出的图像
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                        io = iou(crop_box, np.array(boxes))[0]
                        if io > 0.6:
                            positive_file.write("positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(count,1,offset_x1,
                            offset_y1,offset_x2,offset_y2))
                            face_resize.save(os.path.join(positive_dir, "{0}.jpg".format(count)))
                            count += 1
                        elif io > 0.4:
                            part_file.write("part/{0}.jpg {1} {2} {3} {4} {5}\n".format(coun, 2, offset_x1,
                                                                                                offset_y1, offset_x2,
                                                                                                offset_y2))
                            face_resize.save(os.path.join(part_dir, "{0}.jpg".format(coun)))
                            coun += 1
                        # elif io <= 0.29:
                        #     face_resize.save(os.path.join(negative_dir, "{0}.jpg".format(count)))
                        #     cou += 1
                        _boxes = np.array(boxes)
                        for i in range(5):  # 数量一般和前面保持一样
                            side_len = np.random.randint(face_size, min(img_w, img_h)/2)

                            x_ = np.random.randint(0, img_w - side_len)
                            y_ = np.random.randint(0,img_h - side_len)
                            crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                            if iou(crop_box, _boxes) < 0.005:  # 在加IOU进行判断：保留小于0.3的那一部分；原为0.3
                                negative_file.write(
                                    "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(cou, 0))
                                face_crop = img.crop(crop_box)  # 抠图
                                face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)  # ANTIALIAS：平滑,抗锯齿
                                face_resize.save(os.path.join(negative_dir, "{0}.jpg".format(cou)))
                                cou += 1
            except Exception as e:
                traceback.print_exc()  # 如果出现异常，把异常打印出来
    finally:
        positive_file.close() #关闭正样本txt件
        negative_file.close()
        part_file.close()
