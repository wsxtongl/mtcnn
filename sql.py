import torch
import torch.nn as nn
import numpy as np
import pymysql


class Connect():
    def __init__(self):
        self.connect = pymysql.connect(          #连接
            host="localhost",
            user="root",
            password="root",
            database="test",
            port=3306,
            charset="utf8")

    def check_sql(self,args):
        dirty_stuff = ["\"", "\\", "/", "*", "'", "=", "-", "#", ";", "<", ">", "+", "%", "$", "(", ")", "%", "@", "!"]
        new_args = []
        for arg in args:
            if type(arg) == int or type(arg) == float:
                new_args.append(arg)
                continue
            for stuff in dirty_stuff:
                if arg.__contains__(stuff):
                    arg = arg.replace(stuff, "")
            new_args.append(arg)
        return new_args
    def save_features(self,sql_str,args=None):
        if args is None:
            args = []
        #args = self.check_sql(args)
        #print(f'execute sql : [{sql_str}]   args: [{args}]')
        cursor_test = self.connect.cursor()      #获取游标
        try:
            cursor_test.execute(sql_str, tuple(args))
            self.connect.commit()
        except Exception as e:
            self.connect.rollback()
            print(e)
        finally:
            cursor_test.close()
    def save_faceData(self, name, feature):
        self.save_features("insert into face(name,feature) values(%s,%s)", (name, feature))
    def load_faceData(self):
        face_encod = []
        face_name = []
        cursor_test = self.connect.cursor()
        cursor_test.execute("select * from face")

        #cursor_test.execute("select count(1) from face")
        data_result = cursor_test.fetchall()
        for row in data_result:
            name = row[0]
            face_encoding_str = row[1]
            face_encod.append(decoding_FaceStr(face_encoding_str))
            face_name.append(name)
        return face_name,face_encod

def decoding_FaceStr(encoding_str):
    # print("name=%s,encoding=%s" % (name, encoding))
    # 将字符串转为numpy ndarray类型，即矩阵
    # 转换成一个list

    encoding_str = encoding_str.replace('[', '')
    encoding_str = encoding_str.replace(']', '')
    encoding_str = encoding_str.replace('\n', '')
    encoding_str = encoding_str.replace(',', ' ')

    #print(encoding_str)
    dlist = encoding_str.strip().split()

    # 将list中str转换为float
    np_dlist = np.array(dlist)
    face_encoding =np_dlist.astype(np.float).reshape(-1,1000)

    return face_encoding
def encoding_FaceStr(image_face_encoding):
    # 将numpy array类型转化为列表
    encoding_str_list = [str(i) for k in image_face_encoding for i in k]

    # 拼接列表里的字符串
    encoding_str = ','.join(encoding_str_list)
    return encoding_str

if __name__ == '__main__':
     connect = Connect()
     data = connect.load_faceData()
     print(data[0])
    #pass


# connect = pymysql.connect(          #连接
#             host="localhost",
#             user="root",
#             password="root",
#             database="test",
#             port=3306,
#             charset="utf8")
# sql = """CREATE TABLE face (
#          NAME  CHAR(20) NOT NULL,
#          feature TEXT )"""
# sq = """INSERT INTO user1(FIRST_NAME,
#          LAST_NAME, AGE, SEX, INCOME)
#          VALUES ('Fei', 'Fei', 20, 'M', 1000)"""
#
# s = """
#     select * from user1"""
# sqll = "delete  from face"
#
#     # 执行 sql 语句
# cursor_test = connect.cursor()
# cursor_test.execute(sql)
    # 显示出所有数据
#     data_result = cursor_test.fetchall()
#     print(data_result[:])
#     for row in data_result:
#         fname = row[0]
#         lname = row[1]
#         age = row[2]
#         sex = row[3]
#         income = row[4]
#         # 打印结果
#         print("fname=%s,lname=%s,age=%s,sex=%s,income=%s" % \
#               (fname, lname, age, sex, income))
# except:
#     print("Error: unable to fetch data")
#     cursor_test.execute(sq)
# connect.commit()


# def compare(face1,face2):
#     face1_norm = nn.functional.normalize(face1,dim=0)
#     face2_norm = nn.functional.normalize(face2,dim=0)
#     casa = torch.matmul(face1_norm,face2_norm.t())
#     return casa
# a = torch.Tensor([1,2,])
# b = torch.Tensor([-1,-2])
# print(compare(a,b))

# a = torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float32)
# b = torch.Tensor([1,0,1,1])
#
# c  = a.index_select(dim=0, index=b.long())
# count = torch.histc(b, 2, min=0, max=1)
# count_exp = count.index_select(dim=0,index=b.long())
# print(count)
# print(count_exp)
# #print(torch.pow(a,2).sum(dim=1,keepdim=True).expand(3,2))
# print(c)
#
# b = torch.Tensor([[1,2]])
# print(torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(a-b,2),dim=1)),2)))
# print(torch.sum(torch.sum((a-b) ** 2, dim=1) / 2.0 / 2))
#loss1 = torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(2,xs-self.center_exp),dim=1)),self.count_exp))