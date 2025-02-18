import os
import sys
import json
import pickle
import random
import math

import torch
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
from metrics import Measurement
from sklearn.metrics import precision_score, recall_score, f1_score


def already_split_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_path = os.path.join(root, 'Train')
    val_path = os.path.join(root, 'Val')

    # class_list = [cla for cla in os.listdir(train_path) if os.path.isdir(os.path.join(root, cla))]
    class_list = os.listdir(train_path)
    class_list.sort()
    class_indices = dict((k, v) for v, k in enumerate(class_list))      # class 들이 번호가 매겨져 저장
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in class_list:  # cla 는 class_list 의 하나 하나
        train_cla = os.path.join(train_path, cla)   # train_cla = .root/Training/class
        train_images = [os.path.join(train_path, cla, i) for i in os.listdir(train_cla)   # os.listdir = 한 class 에 있는 모든 image 들, i = 이미지 하나
                        if os.path.splitext(i)[-1] in supported]    # train_images = Training/class/image ...
        train_images.sort()
        image_class = class_indices[cla]    # cla 에 할당된 class 에 대한 라벨 번호를 저장
        every_class_num.append(len(train_images))   # every_class_num = 이미지 개수

        val_cla = os.path.join(val_path, cla)   # val_cla = root/Validation/class
        val_images = [os.path.join(val_path, cla, i) for i in os.listdir(val_cla)
                      if os.path.splitext(i)[-1] in supported]
        val_images.sort()

        for img_path in train_images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)
        for img_path in val_images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_split_data(root: str, val_rate: float = 0.1):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    # flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class = os.listdir(root)
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:        # flower_class = class 가 list 형태로 담겨저 있음
        cla_path = os.path.join(root, cla)      # 전체 = /Field Plant/class/image, root = /F P, cla_path = /F P/class
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)  # os.listdir(cla_path) = image 전체
                  if os.path.splitext(i)[-1] in supported]  # images = /F P/class/image 하나, for 문에 의해 돌아감
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]    # class 하나
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:     # images 는 이미지 경로들, img_path 는 그 중에 하나 뽑은 거
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def rgb_to_lab_opencv(tensor_img):
    # PyTorch Tensor (B, C, H, W) → NumPy 배열 (B, H, W, C)
    np_img = tensor_img.permute(0, 2, 3, 1).cpu().numpy()

    # OpenCV로 RGB → LAB 변환 (배치 단위 처리)
    lab_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in np_img])

    # NumPy (B, H, W, C) → PyTorch Tensor (B, C, H, W)
    lab_tensors = torch.tensor(lab_imgs, dtype=torch.float32).permute(0, 3, 1, 2)

    return lab_tensors


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        if type(model).__name__ == "TwoPathInceptionV3":
            print("TwoPathInceptionV3")
            lab_img = rgb_to_lab_opencv(images)
            l_img = lab_img[:, :1, :, :]  # L 채널 (B, 1, H, W)
            ab_img = lab_img[:, 1:, :, :]  # AB 채널 (B, 2, H, W)
            pred = model(l_img.to(device), ab_img.to(device))

        else:
            pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        # lr_scheduler.step()

    train_loss = accu_loss.item() / (step + 1)
    train_acc = accu_num.item() / sample_num

    return train_loss, train_acc


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num_classes):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    class_correct = torch.zeros(num_classes).to(device)  # zeros 에 class 개수 넣기
    class_total = torch.zeros(num_classes).to(device)
    class_per_accuracy = torch.zeros(num_classes).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    measurement = Measurement(num_classes)
    total_acc, total_precision, total_recall, total_f1score = 0, 0, 0, 0

    for step, data in enumerate(data_loader):
        images, labels = data
        batch_size = images.shape[0]
        sample_num += batch_size   # total data num

        if type(model).__name__ == "TwoPathInceptionV3":
            lab_img = rgb_to_lab_opencv(images)
            l_img = lab_img[:, :1, :, :]  # L 채널 (B, 1, H, W)
            ab_img = lab_img[:, 1:, :, :]  # AB 채널 (B, 2, H, W)
            pred = model(l_img.to(device), ab_img.to(device))

        else:
            pred = model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        correct = torch.eq(pred_classes, labels.to(device))
        accu_num += correct.sum()   # total correct

        # if epoch == 1:
        #     pred_cpu, label_cpu = pred_classes.cpu().numpy(), labels.to(device).cpu().numpy()
        #     # _acc, _precision, _recall, _f1score = measurement(pred_cpu, label_cpu)
        #     _precision = precision_score(label_cpu, pred_cpu, average='micro')
        #     _recall = recall_score(label_cpu, pred_cpu, average='micro')
        #     _f1score = f1_score(label_cpu, pred_cpu, average='micro')

        # # total_acc += _acc * batch_size  # 배치 별 metric 이기 때문에
        # total_precision += _precision * batch_size  # batch_size 를 곱해 줘야
        # total_recall += _recall * batch_size    # sample_num(total) 로 나눴을 때
        # total_f1score += _f1score * batch_size  # 계산이 올바름

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

        class_accuracy = class_correct / class_total    # class 별 acc
        average_accuracy = torch.mean(class_accuracy)

        if epoch == 1:
            data_loader.desc = "[ test ] loss: {:.3f}, acc: {:.3f}".format(
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num    # total correct / total data num = accuracy
            )

        else:
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num
            )

    for i in range(num_classes):
        class_per_accuracy[i] = class_correct[i] / class_total[i]

    val_loss = accu_loss.item() / (step + 1)
    val_acc = accu_num.item() / sample_num

    # total_acc = total_acc / sample_num
    # total_precision = total_precision / sample_num
    # total_recall = total_recall / sample_num
    # total_f1score = total_f1score / sample_num

    # return val_loss, val_acc, average_accuracy.item(), class_per_accuracy, total_acc, total_precision, total_recall, total_f1score
    return val_loss, val_acc, average_accuracy.item(), class_per_accuracy


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
