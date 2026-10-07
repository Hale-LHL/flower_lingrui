import os

from PIL import Image
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

from src.config import CLASSES, DATA_DIR

class Mydataset(Dataset):
    def __init__(self, root_dir, kind_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.kind_dir = kind_dir
        self.label_dir = label_dir
        self.transform = transform
        self.path = os.path.join(self.root_dir, self.kind_dir, self.label_dir)
        self.img_path = [f for f in os.listdir(self.path) if f.lower().endswith((".jpg"))]#题中图片默认格式为jpg
        #.lower().endswith()分别是转小写，看后缀

    def __getitem__(self, idx):
        img_idx_name = self.img_path[idx]
        img_idx_path = os.path.join(self.root_dir, self.kind_dir, self.label_dir, img_idx_name)
        img = Image.open(img_idx_path).convert("RGB")#.convert("RGB")强制转成三通道，不转的话灰度图是 1 通道，Conv2d(3,...) 会崩
        if self.transform is not None:#不能写 != None  #便于还没写transform时的调试
            img = self.transform(img)
        label = CLASSES.index(self.label_dir)#CLASSES.index("daisy") = 0
        return img, label

    def __len__(self):
        return len(self.img_path)


def build_dataset(split, transform=None, data_dir=DATA_DIR):
    return ConcatDataset([Mydataset(data_dir, split, c, transform) for c in CLASSES])
#一次性造出某个 split（train 或test）的完整数据集:把 5 个类别各自的 Mydataset拼成一个大的(dataloader拿到的时候拿的是一个整体)
#参数顺序为 (split, transform=None, data_dir=DATA_DIR)：把最常用的 transform放第二位
#之前写成 (split, data_dir, transform) ，调用 build_dataset("train", train_tf) 会把 transform 误当成 data_dir 传进去


