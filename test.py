import os
import json
import torch
import pandas as pd
import numpy as np
from ADCCPANet import  MT
from p2twindowdwattn import p2t_tiny

from scipy import ndimage
from torchvision import transforms
import nibabel as nib  # nii格式一般都会用到这个包

import skimage

def normalize(volume):
    """归一化"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img, h=128, w=128, d=128):
    """修改图像大小"""
    # # Get current depth
    current_depth = img.shape[-1]
    # 旋转
    img = ndimage.rotate(img, 90, reshape=False)
    # 数据调整
    if current_depth <= d:
        img = skimage.transform.resize(img,[h, w, d],order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)
    else:
        idx = np.random.choice(range(img.shape[-1]), d)
        img = img[:, :, idx]
    return img.astype(np.float32)

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    
    image_size=224
    d=64
    # create model
    # model = MT(image_size=image_size, in_channels=d, num_classes=2).to(device)
    model = p2t_tiny(img_size=image_size, in_chans=d, num_classes=2).to(device)

    # load model weights
    model_weight_path = "./weights/37.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    root = "./TestData/"
    file_list = os.listdir(root)  #返回当前文件夹里包含的文件或者文件夹的列表
    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(int(image_size * 1.143)),
                                   transforms.CenterCrop(image_size)])
    
    print("Testing...")

    # pred = None
    # for _ in range(10):
        
    #         print(filename, predict)
    #     prelist = np.vstack(predict)
    #     print(prelist)
    #     if pred is None:
    #         pred = prelist
    #     else:
    #         pred += prelist
    predict = []
    class_list = []
    for filename in file_list:
        print(filename)
        filepath = os.path.join(root, filename)
        nii_volume = nib.load(filepath)  # 读取nii
        volume = nii_volume.get_fdata().squeeze()
        # 归一化
        # volume = normalize(volume)
        volume = resize_volume(volume, h=image_size, w=image_size, d=d)
        img = test_transform(volume).unsqueeze(0)
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            class_list.append(class_indict[str(predict_cla)])
    print("Creating csv")
    # 生成提交结果的DataFrame，其中包括样本ID和预测类别。
    submit = pd.DataFrame(
        {
            'uuid': [int(x[:-4]) for x in file_list],   # 提取测试集文件名中的ID
            'label': class_list,                        # 预测的类别
        }
    )
    # submit['label'] = submit['label'].map({1:'NC', 0: 'MCI'})
    # 按照ID对结果排序并保存为CSV文件
    submit = submit.sort_values(by='uuid')
    submit.to_csv('submit.csv', index=None)
    print("Done!")

if __name__ == '__main__':
    main()