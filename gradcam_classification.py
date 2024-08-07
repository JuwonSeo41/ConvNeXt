from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import convnext_small, MobileNetV2, ResNet50,VGG16
import os
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np

class_list = {
        "AP_C": 0, "AP_H": 1, "AP_R": 2, "AP_S": 3, "BB_H": 4, "CH_H": 5, "CH_P": 6, "CO_B": 7, "CO_C": 8, "CO_H": 9, "CO_R": 10,
        "GR_B": 11, "GR_E": 12, "GR_H": 13, "GR_R": 14, "OR_H": 15, "PB_B": 16, "PB_H": 17, "PH_B": 18, "PH_H": 19, "PO_EB": 20,
        "PO_H": 21, "PO_LB": 22, "RE_H": 23, "SO_H": 24, "SQ_H": 25, "ST_H": 26, "ST_S": 27, "TO_BS": 28, "TO_EB": 29, "TO_H": 30,
        "TO_LB": 31, "TO_M": 32, "TO_SL": 33, "TO_SS": 34, "TO_TS": 35, "TO_V": 36, "TO_Y": 37
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = VGG16(38)
model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/ConvNeXt/weights/1st_VGG16_batch8.pth'))
model.eval()
model.to(device)

# target_layer = [model.stages[-1][-1].dwconv]  # convnext_small
target_layer = [model.features[2]]   # VGG16
# target_layer = [model.features[0][0]]   # MobileNetV2
# target_layer = [model.layer2[-1].conv3]   # ResNet50
# class_index = 0
# targets = [ClassifierOutputTarget(class_index)]       # 주의시킬 class index

print(target_layer)

img_path = '/content/blurred/PV/1st/Train'
imgs = []
for class_folder in os.listdir(img_path):
    class_path = os.path.join(img_path, class_folder)
    imgs_name = os.listdir(class_path)
    for img in imgs_name:
        imgs.append(os.path.join(class_path, img))  # 이미지들의 경로가 리스트로 담김

imgs.sort()

for image in imgs:
    filename = os.path.split(image)[-1]
    class_name = os.path.split(os.path.split(image)[0])[-1]
    class_index = class_list.get(class_name)
    targets = [ClassifierOutputTarget(class_index)]

    rgb_image = Image.open(image).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(rgb_image).unsqueeze(0).to(device)

    print(f"Input tensor size: {input_tensor.size()}")

    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True, aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]   # 차원 제거

    print(f"Grayscale CAM size: {grayscale_cam.shape}")

    image = np.array(rgb_image) / 255.0
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    visualization = Image.fromarray(visualization)
    grayscale_cam = (grayscale_cam * 255).astype(np.uint8)
    grayscale_cam = Image.fromarray(grayscale_cam)
    os.makedirs(f'/content/drive/MyDrive/Colab Notebooks/ConvNeXt/grad_cam/{model.__class__.__name__}_{target_layer}/gray-cam/{class_name}', exist_ok=True)
    os.makedirs(f'/content/drive/MyDrive/Colab Notebooks/ConvNeXt/grad_cam/{model.__class__.__name__}_{target_layer}/rgb-cam/{class_name}', exist_ok=True)
    grayscale_cam.save(f'/content/drive/MyDrive/Colab Notebooks/ConvNeXt/grad_cam/{model.__class__.__name__}_{target_layer}/gray-cam/{class_name}/{filename}')
    visualization.save(f'/content/drive/MyDrive/Colab Notebooks/ConvNeXt/grad_cam/{model.__class__.__name__}_{target_layer}/rgb-cam/{class_name}/{filename}')
    print(filename)
