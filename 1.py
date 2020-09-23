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
    print(xx2-xx1)
    # 判断是否有交集
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    print(w)
    print(h)
    #交集的面积
    inter = w * h  #对应位置元素相乘
    if isMin: #若果为Ture
        ovr = np.true_divide(inter, np.minimum(box_area, area)) #最小面积的IOU：O网络用
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))  #并集的IOU：P和R网络用；交集/并集

    return ovr
a = np.array([1,2,3,4])
b = np.array([[1,2,3,4],[5,6,7,8]])

io = iou(a,b)
print(io)