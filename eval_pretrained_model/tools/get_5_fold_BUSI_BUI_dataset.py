import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
import random
import cv2
from PIL import Image
import random
import pickle

label_name = {"Ben": 0, "Mal": 1, "Nor": 2}

# train_data_path = "F:/USCL/8-COVID-Xray-5k/COVID-Xray-5k/train/"
# output_train_data_path = "F:/USCL/8-COVID-Xray-5k/COVID_Xray_5k_CL/"
# if not os.path.exists(output_train_data_path):
#     os.mkdir(output_train_data_path)

# test_data_path = "F:/USCL/8-COVID-Xray-5k/COVID-Xray-5k/test/"
# output_test_data_path = "F:/USCL/8-COVID-Xray-5k/covid_xray_5k_5_fold/"
# if not os.path.exists(output_test_data_path):
#     os.mkdir(output_test_data_path)

test_data_path = "/Data/Medical_Image_Datasets/test/BUSI_BUI"
output_test_data_path = "/Data/Medical_Image_Datasets/test/BUSI_BUI_5_fold"
if not os.path.exists(output_test_data_path):
    os.mkdir(output_test_data_path)

# ####################################################################################
# ## Step1. 制作符合对比学习的训练数据
# train_sub_path = os.listdir(train_data_path)
# for sub in train_sub_path:
#     if sub == "covid":
#         label = "Cov"
#     else:
#         label = "Non"
#     train_images = os.listdir(train_data_path+sub)
#     for image_name in train_images:
#         old_path = os.path.join(train_data_path, sub) + "/" + image_name
#
#
#         new_image_name = label + "-"+ image_name.replace(".jpeg", ".jpg").replace(".png", ".jpg")
#         if not os.path.exists(os.path.join(output_train_data_path, "train") + "/" + new_image_name[0:-4]):
#             os.mkdir(os.path.join(output_train_data_path, "train") + "/" + new_image_name[0:-4])
#         else:
#             print(os.path.join(output_train_data_path, "train") + "/" + new_image_name[0:-4])
#
#         new_path = os.path.join(output_train_data_path, "train") + "/" + new_image_name[0:-4]+"/"+new_image_name
#         shutil.copy(old_path, new_path)
#
# print("Train, OK")


####################################################################################
## Step2. 制作5折交叉验证的测试数据

test_sub_path = os.listdir(test_data_path)
temp_path = []
for sub in test_sub_path:
    if sub == "benign":
        label = "Ben"

        test_images = os.listdir(os.path.join(test_data_path, sub))
        N = 0
        total = len(test_images)
        random.shuffle(test_images)   # 随机打乱图片顺序
        for image_name in tqdm(test_images):
            path = os.path.join(test_data_path, sub, image_name)
            print(N, path)
            # path = os.path.join(test_data_path, sub) + "/" + image_name.replace(".jpeg", ".jpg").replace(".png", ".jpg")
            if N<int(total*0.2):
                ss = [path, label, str(0)]
                temp_path.append(ss)
            elif N>=int(total*0.2) and N<int(total*0.4):
                ss = [path, label, str(1)]
                temp_path.append(ss)
            elif N>=int(total*0.4) and N<int(total*0.6):
                ss = [path, label, str(2)]
                temp_path.append(ss)
            elif N>=int(total*0.6) and N<int(total*0.8):
                ss = [path, label, str(3)]
                temp_path.append(ss)
            else:
                ss = [path, label, str(4)]
                temp_path.append(ss)
            # print(ss)
            N +=1

    if sub == "malignant":
        label = "Mal"
        # test_subsub_path = os.listdir(os.path.join(test_data_path, sub))
        #
        # for subsub in test_subsub_path:
        test_images = os.listdir(os.path.join(test_data_path, sub))
        N = 0
        total = len(test_images)
        random.shuffle(test_images)  # 随机打乱图片顺序

        for image_name in tqdm(test_images):
            # path = os.path.join(test_data_path, sub, image_name, image_name.replace("_png", ".png"))
            path = os.path.join(test_data_path, sub, image_name)
            print(N, path)
            # path = os.path.join(test_data_path, sub, subsub) + "/" + image_name.replace(".jpeg", ".jpg").replace(".png", ".jpg")
            if N < int(total * 0.2):
                ss = [path, label, str(0)]
                temp_path.append(ss)
            elif N >= int(total * 0.2) and N < int(total * 0.4):
                ss = [path, label, str(1)]
                temp_path.append(ss)
            elif N >= int(total * 0.4) and N < int(total * 0.6):
                ss = [path, label, str(2)]
                temp_path.append(ss)
            elif N >= int(total * 0.6) and N < int(total * 0.8):
                ss = [path, label, str(3)]
                temp_path.append(ss)
            else:
                ss = [path, label, str(4)]
                temp_path.append(ss)
            # print(ss)
            N += 1

    if sub == "normal":
        label = "Nor"
        # test_subsub_path = os.listdir(os.path.join(test_data_path, sub))
        #
        # for subsub in test_subsub_path:
        test_images = os.listdir(os.path.join(test_data_path, sub))
        N = 0
        total = len(test_images)
        random.shuffle(test_images)  # 随机打乱图片顺序

        for image_name in tqdm(test_images):
            # path = os.path.join(test_data_path, sub, image_name, image_name.replace("_png", ".png"))
            path = os.path.join(test_data_path, sub, image_name)
            print(N, path)
            # path = os.path.join(test_data_path, sub, subsub) + "/" + image_name.replace(".jpeg", ".jpg").replace(".png", ".jpg")
            if N < int(total * 0.2):
                ss = [path, label, str(0)]
                temp_path.append(ss)
            elif N >= int(total * 0.2) and N < int(total * 0.4):
                ss = [path, label, str(1)]
                temp_path.append(ss)
            elif N >= int(total * 0.4) and N < int(total * 0.6):
                ss = [path, label, str(2)]
                temp_path.append(ss)
            elif N >= int(total * 0.6) and N < int(total * 0.8):
                ss = [path, label, str(3)]
                temp_path.append(ss)
            else:
                ss = [path, label, str(4)]
                temp_path.append(ss)
            # print(ss)
            N += 1

f1, f2, f3, f4, f5 = 0,0,0,0,0
for i in range(len(temp_path)):
    if temp_path[i][2] == "0":
        f1 +=1
    elif temp_path[i][2] == "1":
        f2 +=1
    elif temp_path[i][2] == "2":
        f3 +=1
    elif temp_path[i][2] == "3":
        f4 +=1
    elif temp_path[i][2] == "4":
        f5 +=1

print("Test, OK")
print(f1, f2, f3, f4, f5)


##########################################################################
## Step3. 制作五折交叉验证的pkl文件
Total_images = len(temp_path)
Total_train = np.zeros(5)
Total_valid = np.zeros(5)

for i in range(len(temp_path)):
    if temp_path[i][2] == '0':
        Total_valid[0] += 1

    if temp_path[i][2] == '1':
        Total_valid[1] += 1

    if temp_path[i][2] == '2':
        Total_valid[2] += 1

    if temp_path[i][2] == '3':
        Total_valid[3] += 1

    if temp_path[i][2] == '4':
        Total_valid[4] += 1
    Total_train = Total_images - Total_valid

images = np.empty((Total_images, 3, 224, 224))
labels = np.empty((Total_images))

## fold_N = 0,1,2,3,4
for fold_N in range(len(Total_train)):
    images_train = np.empty((int(Total_train[fold_N]), 3, 224, 224))
    labels_train = np.empty((int(Total_train[fold_N])))
    images_valid = np.empty((int(Total_valid[fold_N]), 3, 224, 224))
    labels_valid = np.empty((int(Total_valid[fold_N])))

    k, t_k, v_k = 0, 0, 0
    for i in range(len(temp_path)):
        # val
        if temp_path[i][2] == str(fold_N):
            img_path = temp_path[i][0]
            # img = cv2.imread(img_path)
            img = Image.open(img_path) # RGB
            # img = cv2.imread(img_path, 0)
            img = np.array(img)

            try:
                h, w, c = np.shape(img)
                if c > 3:
                    temp = img[:, :, :3]  # 四通道图像去掉图像的透明度通道
                    img = temp
            except:
                h, w = np.shape(img)
                img = np.expand_dims(img, axis=2).repeat(3, axis=2)  # 单通道灰度图转三通道

            images_valid[v_k] = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC).transpose([2, 1, 0])  # [W H C] -- > [C H W]
            labels_valid[v_k] = label_name[temp_path[i][1]]
            v_k += 1

        # training
        if temp_path[i][2] != str(fold_N):
            img_path = temp_path[i][0]
            # img = cv2.imread(img_path)
            img = Image.open(img_path)  # RGB
            img = np.array(img)

            try:
                h, w, c = np.shape(img)
                if c>3:
                    temp = img[:, :, :3]  # 四通道图像去掉图像的透明度通道
                    img = temp
            except:
                h, w = np.shape(img)
                img = np.expand_dims(img, axis=2).repeat(3, axis=2)  # 单通道灰度图转三通道

            images_train[t_k] = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC).transpose([2, 1, 0])  # [W H C] -- > [C H W]
            labels_train[t_k] = label_name[temp_path[i][1]]
            t_k += 1

        k += 1
        print(k)

    images_train = images_train.astype(np.int)
    labels_train = labels_train.astype(np.int)
    images_valid = images_valid.astype(np.int)
    labels_valid = labels_valid.astype(np.int)

    if not os.path.exists(os.path.join("/Data/Medical_Image_Datasets/test/BUSI_BUI_5_fold")):
        os.mkdir("/Data/Medical_Image_Datasets/test/BUSI_BUI_5_fold")
    write_file = open('/Data/Medical_Image_Datasets/test/BUSI_BUI_5_fold/busi_bui_5k_data{}.pkl'.format(fold_N+1), 'wb')
    pickle.dump([images_train, labels_train, images_valid, labels_valid], write_file)  # save pickle
    write_file.close()

