import os
from shutil import copyfile
from sklearn.model_selection import train_test_split


def main():
    data_path = "C:/Users/Seo/Desktop/datasets/CompleteDataset PlantVillage/Images"
    assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

    class_list = os.listdir(data_path)
    img_list = []
    label_list = []
    print(class_list)

    for i in class_list:
        cla_path = os.path.join(data_path, i)
        for img in os.listdir(cla_path):
            img_list.append(os.path.join(cla_path, img))
            label_list.append(i)

    train_img, test_img, train_label, test_label = \
        train_test_split(img_list, label_list, test_size=0.1, stratify=label_list)
    print(train_img)

    valid_img, test_img, valid_label, test_label = \
        train_test_split(test_img, test_label, test_size=0.3, stratify=test_label)
    # Train 90%, Valid 7%, Test 3%

    train_folder = os.path.join('C:/Users/Seo/Desktop/datasets/CompleteDataset PlantVillage', "Train")
    os.makedirs(train_folder, exist_ok=True)
    for img_path, class_label in zip(train_img, train_label):
        class_folder = os.path.join(train_folder, str(class_label))
        os.makedirs(class_folder, exist_ok=True)
        copyfile(img_path, os.path.join(class_folder, os.path.basename(img_path)))

    valid_folder = os.path.join('C:/Users/Seo/Desktop/datasets/CompleteDataset PlantVillage', "Valid")
    os.makedirs(valid_folder, exist_ok=True)
    for img_path, class_label in zip(valid_img, valid_label):
        class_folder = os.path.join(valid_folder, str(class_label))
        os.makedirs(class_folder, exist_ok=True)
        copyfile(img_path, os.path.join(class_folder, os.path.basename(img_path)))

    test_folder = os.path.join('C:/Users/Seo/Desktop/datasets/CompleteDataset PlantVillage', "Test")
    os.makedirs(test_folder, exist_ok=True)
    for img_path, class_label in zip(test_img, test_label):
        class_folder = os.path.join(test_folder, str(class_label))
        os.makedirs(class_folder, exist_ok=True)
        copyfile(img_path, os.path.join(class_folder, os.path.basename(img_path)))


if __name__ == "__main__":
    main()
