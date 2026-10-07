import os
import shutil

import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision import transforms

from src.config import BAD_IMG, DATA_DIR, IMG_SIZE, LOGS_DIR, SEED, TEST_DIR, TRAIN_DIR, VAL_SIZE

# ============================ 图像预处理（两种 transform） ============================
# transforms.Compose() 是打包，执行时按列表顺序依次执行
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),# 50% 概率把图左右镜像
    transforms.RandomRotation(20),#随机选一个角度旋转，范围是-20 ~20
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),#调亮度，对比度，饱和度，最多 ±20%
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),#先转张量，再归一（mean和std是公认的常量）
])

eval_tf = transforms.Compose([#不进行数据增强
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================ 坏图检测 ============================
def handle_bad_img(logs_dir=LOGS_DIR, bad_img=BAD_IMG, train_dir=TRAIN_DIR, test_dir=TEST_DIR):
    os.makedirs(logs_dir, exist_ok=True)
    bad_img = open(bad_img, 'w', encoding='utf-8')
    # 'w' 为每次清空重写，'a' 为追加记录；open() 的返回值必须用一个对象接住
    # encoding='utf-8'：写中文用哪套编码，读的时候也得用同一套，否则 Windows 用默认的 GBK 会乱码
    all_img_sort_dir = [test_dir, train_dir]
    bad = []
    bad_img.write('以下为坏图：\n')
    for img_sort_dir in all_img_sort_dir:
        for img_sort in os.listdir(img_sort_dir):  # listdir 不接受列表
            img_sort_path = os.path.join(img_sort_dir, img_sort)
            if not os.path.isdir(img_sort_path):  # 不是文件夹就跳过
                continue
            for img in os.listdir(img_sort_path):  # 还要再进一层类别（sort）文件夹
                img_path = os.path.join(img_sort_path, img)  # listdir 只返回文件名，必须拼回完整路径
                try:
                    img = Image.open(img_path)
                    img.verify()  # 只读文件头，快，但抓不全所有坏图
                    img = Image.open(img_path)
                    img.load()  # 真正解码像素，能抓出 verify 漏掉的坏图
                    result = True
                except Exception:
                    result = False
                if not result:  # not result ,即 result 为 False 时进行
                    bad.append(img_path)
                    bad_img.write(img_path + '\n')
    bad_img.close()
    if bad != []:
        broken_dir = os.path.join(logs_dir, "broken")
        os.makedirs(broken_dir, exist_ok=True)
        for p in bad:
            # os.path.dirname() 往左砍掉最后一段，返回剩下的
            # os.path.basename() 往右取最后一段
            name = os.path.basename(os.path.dirname(os.path.dirname(p))) + "_" + os.path.basename(os.path.dirname(p)) + "_" + os.path.basename(p)
            dst = os.path.join(broken_dir, name)
            if not os.path.exists(dst):  # 目标位置没有同名文件才移动
                shutil.move(p, dst)  # 剪切，不删除，随时可还原
    return bad


# ============================ 坏图还原(便于编码时看效果) ============================
def restore_bad_img(logs_dir=LOGS_DIR, data_dir=DATA_DIR):
    broken_dir = os.path.join(logs_dir, "broken")
    if not os.path.isdir(broken_dir):
        return
    for name in os.listdir(broken_dir):
        te_or_tr, sort, img_name = name.split("_", 2)#以_为界分开得三个名字。也可以写的时候就写成“路径”的形式，这里就不用换了
        raw_path = os.path.join(data_dir, te_or_tr, sort, img_name)
        if not os.path.exists(raw_path):
            shutil.move(os.path.join(broken_dir, name), raw_path)
    print("损坏图片已还原")



# ============================ 从训练集中划分验证集 ============================
def val_split(dataset, val_size=VAL_SIZE, seed=SEED):
    n = len(dataset)
    n_val = int(round(n * val_size))  # random_split 的长度必须是整数
    n_train = n - n_val  # 用减法保证 n_train + n_val == n，不会漏样本
    gener = torch.Generator().manual_seed(seed)  # 造一个单独的随机数生成器，不受全局种子影响
    train, val = random_split(dataset, [n_train, n_val], generator=gener)
    # 划分是逻辑划分：只记录下标，不复制图片、不新建 val 文件夹（硬盘数据不变）
    return train, val
