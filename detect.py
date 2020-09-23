import torch
import numpy as np
import mynet
from torchvision import transforms
import time
from sample import nms
DEVICE = "cuda:0"
p_cls = 0.6
p_nms = 0.5
class Detector():
    def __init__(self,pnet_param = "./param/pnet.pt",rnet_param = "./param/pnet.pt",onet_param = "./param/pnet.pt"):
        self.pnet = mynet.PNet()
        self.rnet = mynet.RNet()
        self.onet = mynet.ONet()
        self.pnet.to(DEVICE)
        self.rnet.to(DEVICE)
        self.onet.to(DEVICE)
        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))
        self.__img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    def detect(self,img):
        start_time = time.time()
        pnet_box = self.__pnet_detect(img)
        if pnet_box.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        pnet_time = end_time-start_time

        start_time = time.time()
        rnet_box = self.__rnet_detect(img,pnet_box)
        if rnet_box.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        rnet_time = end_time - start_time

        start_time = time.time()
        onet_box = self.__onet_detect(img,rnet_box)
        if onet_box.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        onet_time = end_time - start_time
        return onet_box
    def __pnet_detect(self,img):
        box1 = np.random.randn(1,5)
        w,h = img.size
        min_sidelen = min(w,h)
        scale = 1
        while min_sidelen > 12:
            img_data = self.__img_transform(img)
            img_data.to(DEVICE)
            img_data = torch.unsqueeze(img_data,0)
            _cls,_offset = self.pnet(img_data)
            _cls = _cls[0][0].cpu().detach()
            _offset = _offset[0].cpu().detach()
            index = torch.nonzero(torch.gt(_cls,p_cls))
            box = self.__box(self,index,_offset,_cls[index[:,0],index[:,1]],scale)
            box1 = np.vstack(box1,box)
            scale = scale * 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))
            min_sidelen = min(-w,-h)
        return nms(box1[1:],p_nms)
    def __box(self,index,offset, cls, scale, stride=2, side_len=12):
        _x1 = (cls[:,1] * stride) / scale
        _y1 = (cls[:,0] * stride) / scale
        _x2 = (cls[:,1] * stride + side_len) / scale - 1
        _y2 = (cls[:,0] * stride + side_len) / scale - 1
        ow = _x2 - _x1
        oh = _y1 - _y2
        _offset = offset[:,cls[:,0],cls[:,1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]
        x1 = x1.reshape(-1, 1)
        y1 = x1.reshape(-1, 1)
        x2 = x1.reshape(-1, 1)
        y2 = x1.reshape(-1, 1)
        cls = cls.reshape(-1, 1).float()
        cat = torch.cat((x1,y1,x2,y2,cls),1)
        return cat.numpy()
    def __rnet_detect(self,img,pnet_box):
        pass
    def __onet_detect(self,img,rnet_box):
        pass