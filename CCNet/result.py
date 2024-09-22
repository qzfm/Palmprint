import os
import numpy as np
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T

import argparse
import time
import sys
from tqdm import tqdm

from models.ccnet import ccnet


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):

        # if not T.functional._is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')

        c,h,w = tensor.size()
   
        if c != 1:
            raise TypeError('only support graysclae image.')

        # print(tensor.size)

        tensor = tensor.view(c, h*w)
        idx = tensor > 0
        t = tensor[idx]

        # print(t)
        m = t.mean()
        s = t.std() 
        t = t.sub_(m).div_(s+1e-6)
        tensor[idx] = t
        
        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats = self.outchannels, dim = 0)
    
        return tensor



class MyDataset(data.Dataset):

    def __init__(self, dir, transforms=None, train=True, imside = 128, outchannels = 1):        

        self.train = train

        self.imside = imside # 128, 224
        self.chs = outchannels # 1, 3

        self.img_dir = dir     

        self.transforms = transforms

        if transforms is None:
            if not train: 
                self.transforms = T.Compose([ 
                                                        
                    T.Resize(self.imside),                  
                    T.ToTensor(),   
                    NormSingleROI(outchannels=self.chs)
                    
                    ]) 
            else:
                self.transforms = T.Compose([  
                                
                    T.Resize(self.imside),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),# 0.3 0.35
                        T.RandomResizedCrop(size=self.imside, scale=(0.8,1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),# (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, interpolation=Image.BICUBIC, expand=False, center=(0.5*self.imside, 0.0)),
                            T.RandomRotation(degrees=10, interpolation=Image.BICUBIC, expand=False, center=(0.0, 0.5*self.imside)),
                        ]),
                    ]),     

                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)                   
                    ])

        self._read_txt_file()

    def _read_txt_file(self):

        self.images_path = []

        file_names = os.listdir(self.img_dir)
        file_names.sort()

        for file_name in file_names:
            self.images_path.append(os.path.join(self.img_dir, file_name))

    def __getitem__(self, index):
        """
        参数:
        - index: int,索引值,用于指定要加载的图像数据的位置
        
        返回:
        - data: 经过预处理后的图像数据。
        - file_name: 图像的文件名。
        
        预处理图像数据,然后返回图像数据(tensor)及其文件名(str)
        """
        
        img_path = self.images_path[index]
        file_name = os.path.basename(img_path)
        
        data = Image.open(img_path).convert('L')     
        data = self.transforms(data)    

        return data, file_name
    

    def __len__(self):
        return len(self.images_path)


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    """
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算两个向量的模
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # 计算余弦相似度
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    # 示例中似乎是百分数，所以乘以100
    return similarity * 100


def result(model, data_loader):

    featDB_test = []
    filenames_test = []

    for batch_id, (data, file_names) in enumerate(data_loader):
        data = data.cuda()
        codes = model.getFeatureCode(data)

        codes = codes.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            filenames_test = file_names
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            filenames_test = np.concatenate((filenames_test, file_names)) 

    
    with open("result.txt", "w") as f:

        result_lines = []
        for i in tqdm(range(len(featDB_test))):
            for j in range(len(featDB_test)):

                if i == j:  
                    # 避免自己和自己比对
                    continue

                score = cosine_similarity(featDB_test[i], featDB_test[j])

                result_lines.append(f"{filenames_test[i]} {filenames_test[j]} {score}\n")
        
        print('saving···\n') 
        f.writelines(result_lines)
        print(f"结果已保存至 result.txt")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CO3Net for Palmprint Recfognition"
    )

    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--weight1", type=float, default=0.8)
    parser.add_argument("--weight2", type=float, default=0.2)
    parser.add_argument("--com_weight",type=float,default=0.8)

    parser.add_argument("--id_num", type=int, default=378,
                        help="IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 Tongji_LR 300")
    
    parser.add_argument("--gpu_id", type=str, default='0')

    
    parser.add_argument("--test_set_file", type=str, default='./data/test_list')

    parser.add_argument("--checkpoint", type=str, default='./results/checkpoint/net_params_best.pth')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    batch_size = args.batch_size
    num_classes = args.id_num
    weight1 = args.weight1
    weight2 = args.weight2
    comp_weight = args.com_weight
    checkpoint = args.checkpoint

    test_set_file = args.test_set_file

    test_set = MyDataset(dir=test_set_file, transforms=None, train=False)

    data_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, num_workers=2)
    
    # 推理阶段num_classes无关紧要, 但需要设置和训练阶段相同，避免报错
    model = ccnet(num_classes=600, weight=comp_weight)
    model.cuda()

    model.load_state_dict(torch.load(checkpoint))

    
    result(model, data_loader)

# python result.py --test_set_file H:\dataset\ROI\split33\test\session2 --checkpoint epoch_90_net_params.pth

