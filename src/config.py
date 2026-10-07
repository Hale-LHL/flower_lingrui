import os
import torch

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
LOGS_DIR  = os.path.join(ROOT_DIR, "logs")
BAD_IMG  = os.path.join(LOGS_DIR, "bad_img")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")
VAL_DIR  = os.path.join(DATA_DIR, "val")
ACC_CURVE_PATH = os.path.join(ROOT_DIR, "docx","accuracy_curve")
LOSS_CURVE_PATH = os.path.join(ROOT_DIR, "docx","loss_curve")
BEST_PTH = os.path.join(LOGS_DIR, "best.pth")

CLASSES   = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE  = 227#这里用题中图里的224是不行的，详解见题解
BATCH_SIZE = 32
VAL_SIZE = 0.2
SEED      = 42
EPOCHS = 20
LR = 0.0003
NUM_WORKERS = 2

RESTORE   = True#控制是否进行坏图复原

