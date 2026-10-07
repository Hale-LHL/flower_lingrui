import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   #防止OpenMP打架
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Subset
from src.plot import plot_curves
from src.config import BEST_PTH, CLASSES, DEVICE, LR, RESTORE
from src.dataloader import make_loaders
from src.dataset import build_dataset
from src.model import Mymodel
from src.preproccess import eval_tf, handle_bad_img, restore_bad_img, train_tf, val_split
from src.train import fit
#======================开始工作===============================================
print("==================开始工作==========================")

#=========在调试的时候进行坏图的恢复（放回原位），以便观察坏图的清理是否成功===============
if RESTORE:
    restore_bad_img()

#=========清理坏图===================================================
bad = handle_bad_img()
print("坏图", len(bad), "张")

#==================建立dataset=========================================
train_dataset_tr = build_dataset("train", train_tf)
train_dataset_eval = build_dataset("train", eval_tf)#transform增强后的图根本不会被存下来 —— 它是在取批次的那一瞬间现算现用、用完即弃的。
#以上两个其实一样，只是取用的时候取的数据一个要增强，一个不增强
test_dataset = build_dataset("test", eval_tf)

#===================分val和train的下标====================================
train_part, val_part = val_split(train_dataset_eval)#用train_dataset_tr也行他们两个在dataloader前都一样
val_idx = val_part.indices#indices是取相应的下标
train_idx = train_part.indices
#还有一种方式如下：（这种比较麻烦，但也是一个思路）
#val_idx_set = set(val_idx)  # set就是把下标清单变成方便做判断的形式
#train_idx = [i for i in range(len(train_dataset_tr)) if i not in val_idx_set] #i in range(len(train_dataset_tr))其实就是全部的下标

#===================分val和train的dataset====================================
train_dataset = Subset(train_dataset_tr, train_idx)#从一个已有的 Dataset 里，按下标挑出一部分，包成一个新的 Dataset
val_dataset = Subset(train_dataset_eval, val_idx)
print("各数据集的数据量：train", len(train_dataset), "| val", len(val_dataset), "| test", len(test_dataset))

#===================进行dataloader====================================
train_loader, val_loader, test_loader = make_loaders(train_dataset, val_dataset, test_dataset)

#===================设置模型，损失函数，优化器====================================
model = Mymodel(n_classes=len(CLASSES)).to(DEVICE)#num_classes:分几类，他就是几
loss= nn.CrossEntropyLoss()  # 内部自带 softmax，模型末层不加
opt = optim.Adam(model.parameters(), lr=LR)

#===================进行训练并保存模型====================================
print("正在使用的设备是：", DEVICE)
history, best_val = fit(model, train_loader, val_loader, test_loader, loss, opt)
print(f"最高验证集准确率 {best_val:.4f}")
print("模型已保存", BEST_PTH)

#===================画图并保存====================================
plot_curves(history)
print("已完成所有任务（恭喜恭喜恭喜恭喜~）")

