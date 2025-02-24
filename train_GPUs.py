import timm

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import numpy as np
import random
from config import get_option
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torch.utils.tensorboard import SummaryWriter
import logging
import warnings
from dataset import AiJsonDataset  # 使用的是AI Challenger读取方式
from dataset import BDD100kJsonDataset  # 使用的是BDD100K读取方式

warnings.filterwarnings('ignore')

# 超参数的设置
opt = get_option()

CLASS_NUM = opt.class_num
BATCH_SIZE = opt.batch_size
VAL_BATCH_SIZE = opt.val_batch_size
LOG_PATH = opt.tensorboard_path
EPOCHS = opt.epochs
SEED = opt.seed
IMG_SIZE = opt.img_size
LOG_FILE = opt.log_file
GPUS = opt.GPUS
save_folder = opt.checkpoint_dir

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
file_handler.setFormatter(formatter)

# print to screen
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# 文件路径
# TRAIN_JSON_PATH = 'AI_datasets/scene_train_annotations_20170904.json'
# TRAIN_IMAGE_PATH = 'AI_datasets/train'
#
# VAL_JSON_PATH = 'AI_datasets/scene_validation_annotations_20170908.json'
# VAL_IMAGE_PATH = 'AI_datasets/val'

# 文件路径
TRAIN_JSON_PATH = 'BDD100k-SC/label/train_tian1qi1&shi1jian1.json'
TRAIN_IMAGE_PATH = 'BDD100k-SC/image/train'


VAL_JSON_PATH = 'BDD100k-SC/label/val_tian1qi1&shi1jian1.json'
VAL_IMAGE_PATH = 'BDD100k-SC/image/valids'



# 损失函数的设置
criterion = nn.CrossEntropyLoss()

# 并行训练设置
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

# writer = SummaryWriter(LOG_PATH)

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model(model_choice=0):
    if model_choice == 0:
        # model = mobilenet_v3_small(pretrained=True)
        model = mobilenet_v3_small(weights=True)
        # model = mobilenet_v3_large(weights=True)
        model.classifier[3] = nn.Linear(in_features=1024, out_features=CLASS_NUM, bias=True)
    elif model_choice == 1:
        # model = mobilenet_v3_large(pretrained=True)
        model = mobilenet_v3_large(weights=True)
        model.classifier[3] = nn.Linear(in_features=1280, out_features=CLASS_NUM, bias=True)
    elif model_choice == 2:
        # model = timm.create_model("mobilenetv3_large_100",num_classes=CLASS_NUM,pretrained=True)
        model = timm.create_model("mobilenetv3_large_100", num_classes=CLASS_NUM, weights=True)
    return model


def train(epoch, model):
    train_process = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # train_dataset = AiJsonDataset(TRAIN_JSON_PATH, TRAIN_IMAGE_PATH, train_process)  # 使用的是AI Challenger读取方式
    train_dataset = BDD100kJsonDataset(TRAIN_JSON_PATH, TRAIN_IMAGE_PATH, train_process)  # 使用的是BDD100K读取方式
    # train_dataset = ImageFolder(TRAIN_PATH, transform=train_process)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)

    # optimizer = Adam(model.parameters(), lr=1e-6, betas=[0.9, 0.99], eps=1e-08, weight_decay=0.0)
    optimizer = Adam(model.parameters(), lr=3e-4, betas=[0.9, 0.999], eps=1e-08)
    # optimizer = Adam(model.parameters(), lr=5e-5, betas=[0.9, 0.999], eps=1e-08)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    model.train()
    train_loss = 0
    for image, target in tqdm(train_dataloader):
        # 并行
        # image = image.cuda()
        # target = target.cuda()
        image = image.to(device)
        target = target.to(device)
        predict = model(image)
        loss = criterion(predict, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() * image.size(0)
    train_loss = train_loss / len(train_dataloader.dataset)
    logger.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


def val(epoch, model):
    val_process = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    #val_dataset = JsonDataset(VAL_JSON_PATH, VAL_IMAGE_PATH, transform=val_process)  # 使用的是Place365 json读取方式
    # val_dataset = AiJsonDataset(VAL_JSON_PATH, VAL_IMAGE_PATH, transform=val_process)  # 使用的是AI Challenger读取方式
    val_dataset = BDD100kJsonDataset(VAL_JSON_PATH, VAL_IMAGE_PATH, transform=val_process)  # 使用的是BDD100K读取方式
    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    # val_dataset = MyDataset(VAL_FILE_PATH, VAL_PATH, val_process)
    # val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=8,shuffle=False)

    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in tqdm(val_dataloader):
            # data, label = data.cuda(), label.cuda()
            data, label = data.to(device), label.to(device)

            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            #          writer.add_scalar('Loss/val',loss,epoch)
            val_loss += loss.item() * data.size(0)
    val_loss = val_loss / len(val_dataloader)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    # writer.add_scalar('acc',acc,epoch)
    old_loss, old_acc = 100, 0
    logger.info('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
    # logger.info('\033[42mEpoch: {} \tValidation Loss Reduced: {:.6f}, Accuracy Improved: {:6f}\033[0m'.format(epoch, old_loss - val_loss, acc - old_acc))




if __name__ == '__main__':
    # 设置随机数种子
    seed(SEED)
    # 设置并行训练参数
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    # 加载模型
    # model = get_model(model_choice=1).cuda()
    model = get_model(model_choice=1).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    # model.load_state_dict(torch.load("./checkpoint/MobileNetv3_large_epoch_58.pth")["model_state_dict"])
    print("-------load model-------")
    logger.info('Training batch_size: {} \tVal batch_size: {}'.format(BATCH_SIZE, VAL_BATCH_SIZE))
    logger.info('Training LR: {} \tBetas: {}'.format('5e-5', '0.9-0.999'))
    # 训练数据
    for epoch in range(EPOCHS):
        train(epoch, model)
        val(epoch, model)
        logger.info('----------------------------------')
        if epoch % 5 == 0 and epoch > 0:
            if GPUS > 1:
                checkpoint = {'model': model.module,
                              'model_state_dict': model.module.state_dict(),
                              # 'optimizer_state_dict': optimizer.state_dict(),
                              'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'MobileNetv3_large_epoch_{}.pth'.format(epoch)))
            else:
                checkpoint = {'model': model,
                              'model_state_dict': model.state_dict(),
                              # 'optimizer_state_dict': optimizer.state_dict(),
                              'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'MobileNetv3_large_epoch_{}.pth'.format(epoch)))
                print('MobileNetv3_large_epoch_{}.pth save over'.format(epoch))
