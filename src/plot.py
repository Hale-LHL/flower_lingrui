from src.config import LOSS_CURVE_PATH, ACC_CURVE_PATH, EPOCHS
import matplotlib.pyplot as plt


def plot_curves(history, loss_curve_path =LOSS_CURVE_PATH,acc_curve_path =ACC_CURVE_PATH, epochs=EPOCHS):

    epochs = list(range(1, epochs+1))
    #============================画loss==================================
    plt.figure()
    #marker=""：每个数据点的形状。其他常见的还有 "o" 圆点、"s" 方块、"*" 星形、"^"三角形
    #markersize= ：数据点的大小，默认是 6
    plt.plot(epochs, history["train_loss"], marker="o", markersize=3, label="train")
    plt.plot(epochs, history["val_loss"], marker="s", markersize=3, label="val")
    plt.plot(epochs, history["test_loss"], marker="^", markersize=3, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.legend()
    plt.grid( alpha=0.3)#画网格图

    plt.savefig(loss_curve_path, dpi=120)
    print("loss_curve 已保存:", loss_curve_path)
    plt.close()
    ##============================画accuracy==================================
    plt.plot(epochs, history["train_acc"], marker="o", markersize=3, label="train")
    plt.plot(epochs, history["val_acc"], marker="s", markersize=3, label="val")
    plt.plot(epochs, history["test_acc"], marker="^", markersize=3, label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(acc_curve_path, dpi=150)
    print("accuracy_curve 已保存:", acc_curve_path)
    plt.close()
