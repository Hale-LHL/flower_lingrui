from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DEVICE


def make_loaders(train, val, test, batch_size=BATCH_SIZE,):
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
# shuffle=True 只给训练集：每轮打乱顺序，防止模型记住"顺序"而不是特征
# 验证集和测试集必须 False，顺序固定结果才可复现
