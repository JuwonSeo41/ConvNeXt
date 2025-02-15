import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from my_dataset import MyDataSet
from utils import evaluate

from model import VGG16 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 38

    data_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = "/content/drive/MyDrive/Colab Notebooks/ConvNeXt/weights/1st_VGG16_batch8.pth"      # model종류, weight, test_path 바꾸기!!
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    test_path = '/content/PV_2_fold/1st/Test'
    assert os.path.exists(test_path), "file: '{}' does not exist.".format(test_path)
    test_images_path = []
    test_images_label = []

    test_path_1 = os.listdir(test_path)         # test의 하위 목록에는 class 폴더가 아닌 엑셀파일도 들어있음
    test_path_1.sort()
    class_list = os.listdir('/content/drive/MyDrive/Colab Notebooks/PV_class_list')  # class 이름만 뽑는 것, 경로 바꾸지 않아도 됨
    class_list.sort()
    class_indices = dict((k, v) for v, k in enumerate(class_list))

    for cla in test_path_1:
        file_path = os.path.join(test_path, cla)
        if os.path.basename(file_path) in class_list:       # 엑셀 파일이면 cla_path 에 추가하지 않음
            cla_path = os.path.join(test_path, cla)
            image_class = class_indices[cla]
        img_path = [os.path.join(cla_path, img) for img in os.listdir(cla_path)]
        img_path.sort()

        for i in img_path:
            test_images_path.append(i)
            test_images_label.append(image_class)

    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=8,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=test_dataset.collate_fn)

    test_loss, test_acc, test_average_acc, class_per_accuracy = evaluate(model=model,
                                                                         data_loader=test_loader,
                                                                         device=device,
                                                                         epoch=1)

    class_index = {
        "AP_C": 0, "AP_H": 1, "AP_R": 2, "AP_S": 3, "BB_H": 4, "CH_H": 5, "CH_P": 6, "CO_B": 7, "CO_C": 8, "CO_H": 9,
        "CO_R": 10,
        "GR_B": 11, "GR_E": 12, "GR_H": 13, "GR_R": 14, "OR_H": 15, "PB_B": 16, "PB_H": 17, "PH_B": 18, "PH_H": 19,
        "PO_EB": 20,
        "PO_H": 21, "PO_LB": 22, "RE_H": 23, "SO_H": 24, "SQ_H": 25, "ST_H": 26, "ST_S": 27, "TO_BS": 28, "TO_EB": 29,
        "TO_H": 30,
        "TO_LB": 31, "TO_M": 32, "TO_SL": 33, "TO_SS": 34, "TO_TS": 35, "TO_V": 36, "TO_Y": 37
    }

    index_to_class = {v: k for k, v in class_index.items()}

    testtxt = '\n\ttest loss: {:.6f}\ttest accuracy: {:.6f}\ttest average accuracy: {:.6f}\tmodel: {}'\
        .format(test_loss, test_acc, test_average_acc, create_model.__name__)
    cla_per_acc = ['[{}] accuracy: {:.6f}'.format(index_to_class[i], class_per_accuracy[i]) for i in range(num_classes)]
    cla_per_acc_str = '\n'.join(cla_per_acc)     # 리스트를 하나의 문자열로 연결
    print(testtxt)
    print(cla_per_acc)
    save_path = open(os.path.join('C:/Users/Seo/PycharmProjects/pytorch_classification/ConvNeXt', 'testtxt.txt'), "a")
    save_path_1 = open(os.path.join('C:/Users/Seo/PycharmProjects/pytorch_classification/ConvNeXt', 'cla_per_acc.txt'), "a")
    save_path.write(testtxt)
    save_path_1.write(cla_per_acc_str)

if __name__ == '__main__':
    main()
