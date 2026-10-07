import torch

from src.config import BEST_PTH, DEVICE, EPOCHS


def train_one_epoch(model, loader, loss, opt, device):
    model.train()  # 开启 Dropout、BatchNorm 用 batch 统计量
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()        # 1. 清空上一轮的梯度，否则梯度会累加
        out = model(x)        # 2. 前向传播
        loss_val = loss(out, y)     # 3. 算损失
        loss_val.backward()         # 4. 反向传播
        opt.step()            # 5. 更新参数
        total_loss += loss_val.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()  # argmax(1) 取每行最大值的下标即预测类别
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, loss, device):
    model.eval()  # 关掉 Dropout、BatchNorm
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():#进入块前自动关梯度，出来后自动恢复，不用手动开关
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss_value = loss(out, y)#用来装"损失值"的变量，千万别和"损失函数"同名
            total_loss += loss_value.item() * y.size(0)#第一排有多少个数据（y一共有多少数据）
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, test_loader, loss, opt,
        epochs=EPOCHS, device=DEVICE, best_pth=BEST_PTH):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    test_loss, test_acc = [], []
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss, opt, device)
        va_loss, va_acc = evaluate(model, val_loader, loss, device)
        te_loss, te_acc = evaluate(model, test_loader, loss, device)

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        val_loss.append(va_loss)
        val_acc.append(va_acc)
        test_loss.append(te_loss)
        test_acc.append(te_acc)

        print("epoch %2d/%d | train loss %.4f； acc %.4f | val loss %.4f； acc %.4f | test loss %.4f； acc %.4f"
              % (epoch, epochs, tr_loss, tr_acc, va_loss, va_acc, te_loss, te_acc))

        if va_acc > best_val:  # 只按验证集挑最好的一轮保存，不能用测试集挑
            best_val = va_acc
            torch.save(model.state_dict(), best_pth)  # 先只存权重，不存整个模型（不是选做的内容———存模型）

    return {"train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "test_loss": test_loss, "test_acc": test_acc}, best_val
