import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from my_dataset import MyDataSet
from utils import evaluate

from model import convnext_small as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 38
    img_size = 256
    data_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # img_path = "../tulip.jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # plt.imshow(img)
    # # [N, C, H, W]
    # img = data_transform(img)
    # # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = "../ConvNeXt/weights/1st_convnext_small_batch8.pth"      # model종류, weight, test_path 바꾸기!!
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    test_path = '/content/drive/MyDrive/Colab Notebooks/test/PV/WRANet5/restored_imgs'
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

    test_loss, test_acc, test_average_acc = evaluate(model=model,
                                                     data_loader=test_loader,
                                                     device=device,
                                                     epoch=1)

    testtxt = '\n\ttest loss: {:.6f}\ttest accuracy: {:.6f}\ttest average accuracy: {:.6f}\tmodel: {}'\
        .format(test_loss, test_acc, test_average_acc, create_model.__name__)
    print(testtxt)
    save_path = open(os.path.join('C:/Users/Seo/PycharmProjects/pytorch_classification/ConvNeXt', 'testtxt.txt'), "a")
    save_path.write(testtxt)

    # for class_folder in os.listdir(img_folder_path):
    #     class_folder_path = os.path.join(img_folder_path, class_folder)
    #
    #     if os.path.isdir(class_folder_path):
    #         print(f"\nClass: {class_folder}")
    #     for img in os.listdir(class_folder_path):
    #         img_path = os.path.join(class_folder_path, img)
    #         assert os.path.exists(img_path)
    #         img = Image.open(img_path)
    #         plt.imshow(img)
    #         img = data_transform(img)
    #         img = torch.unsqueeze(img, dim=0)  # img 에 batch 차원을 추가해 [B, C, H, W] 로 만듬
    #         with torch.no_grad():
    #             output = torch.squeeze(model(img.to(device))).cpu()     # 이미지를 모델에 넣어서 예측 결과를 나타냄
    #             predict = torch.softmax(output, dim=0)      # 예측 결과에 대한 각 class 의 확률 값으로 저장
    #             predict_cla = torch.argmax(predict).numpy()     # 예측 결과가 가장 높은 class 즉, 이미지에 대해 모델이 예측한 class

    # with torch.no_grad():
    #     # predict class
    #     output = torch.squeeze(model(img.to(device))).cpu()
    #     predict = torch.softmax(output, dim=0)
    #     predict_cla = torch.argmax(predict).numpy()
    #
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # plt.show()


if __name__ == '__main__':
    main()
