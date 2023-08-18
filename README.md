# CV实践打卡笔记一 脑PET分类    

## 题目总结
基于一定数量的脑PET图像完成模型训练，并进行分析和预测。    

## 数据组成
脑PET图像检测数据库，记录了老年人受试志愿者的脑PET影像资料，文件格式为 **nii**。包括确诊为轻度认知障碍（MCI）患者的脑部影像数据和健康人（NC）的脑部影像数据。    

数据集构成：   
- **Train** **50**
    - **NC**    **25**   
    - **MCI**   **25**  
- **Test**  **100**  

## 心得
学会了如何对医学图像进行处理、学会了如何将其他相关任务的模型进行调整，以及在面临一个从未接触过的课题或项目时内心的情绪如何调节管理。
## 数据预处理
使用**nibabel**读取nii格式的脑部PET文件。根据获取到的图片尺寸以及图像 pixdim 信息可知图像所代表实际尺寸多数为 263.6×263.6×152.8，决定将图像尺寸压缩为 224×224×64.其中，通道数 ＞64 的随机选择 64 个通道，＜64 的使用 skimage 进行调整。   
### MyDataSet      
    import torch
    from torch.utils.data import Dataset
    import numpy as np
    from scipy import ndimage
    import nibabel as nib  # nii格式一般都会用到这个包
    import skimage

    def resize_volume(img, h=224, w=224, d=64):
        """修改图像大小"""
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

    class BrainPETSet(Dataset):
        """自定义数据集"""
        def __init__(self, images_path: list, images_class: list, h=128, d=64, transform=None):
            self.images_path = images_path
            self.images_class = images_class
            self.transform = transform
            self.h = h
            self.d = d

        def __len__(self):
            return len(self.images_path)

        def __getitem__(self, item):
            nii_volume = nib.load(self.images_path[item])  # 读取nii
            volume = nii_volume.get_fdata().squeeze()
            # 调整尺寸 h=128, w=128, d=64
            img = resize_volume(volume, h=self.h, w=self.h, d=self.d).astype(np.float32)
            label = self.images_class[item]
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        @staticmethod
        def collate_fn(batch):
            images, labels = tuple(zip(*batch))
            images = torch.stack(images, dim=0)
            labels = torch.as_tensor(labels)
            return images, labels

## 构建模型

基于一种CNN与Transformer相结合的开源模型 **P2TNet** 对任务进行优化。

## 模型训练

### 1. 数据增强及数据集实例化
    img_size = 224
    d = 64
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomResizedCrop(img_size),
                                    transforms.RandomHorizontalFlip()]),
        "val": transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(int(img_size * 1.143)),
                                transforms.CenterCrop(img_size)])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                            images_class=train_images_label,
                            h=img_size,
                            d=d,
                            transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            h=img_size,
                            d=d,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=val_dataset.collate_fn)
### 2. 模型实例化、优化器、学习率调整策略
        model = p2t_tiny(img_size=img_size, in_chans=d, num_classes=2).to(device)

        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                        warmup=True, warmup_epochs=1)
### 3. 开始训练
        best_acc = 0.
        best_f1 = 0.
        best_epoch = 0
        for epoch in range(args.epochs):
            # train
            train_loss, train_acc, train_f1 = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    lr_scheduler=lr_scheduler)
            # validate
            val_loss, val_acc, val_f1 = evaluate(model=model,
                                        data_loader=val_loader,
                                        device=device,
                                        epoch=epoch)
            # 实时显示并记录训练中的各项参数 {epoch train:acc\loss\f1  val:acc\loss\f1}
            output1 = "[epoch {:.0f}] :  train_accuracy:{:.4f}  train_loss:{:.3f}  train_f1:{:.4f}  val_accuracy:{:.4f}  val_loss:{:.3f}  val_f1:{:.4f}".format(
                epoch,
                train_acc,
                train_loss,
                train_f1,
                val_acc,
                val_loss,
                val_f1,)
            with open(file_name, "a+") as f:
                f.write(output1 + '\n')
                f.close
            # 保存模型训练过程中的权重以便复现时比较
            torch.save(model.state_dict(), "./weights/{}.pth".format(epoch))
            # 记录模型最优表现
            if val_acc > best_acc:
                best_acc = val_acc
                best_f1 = val_f1
                torch.save(model.state_dict(), "./weights/best.pth")
                best_epoch = epoch
            elif val_acc == best_acc:
                if val_f1 >= best_f1:
                    best_f1 = val_f1
                    torch.save(model.state_dict(), "./weights/best.pth")
                    best_epoch = epoch
                else:
                    continue
        # 保存模型最优表现
        output2 = "epoch:{},best_acc:{:.4f},best_f1:{:.4f}".format(best_epoch, best_acc, best_f1)
        with open(file_name, "a+") as f:
            f.write(output2 + '\n')
            f.close
## 模型预测
### 1. 创建模型并加载权重
    image_size=224
    d=64
    # create model
    model = p2t_tiny(img_size=image_size, in_chans=d, num_classes=2).to(device)
    # load model weights
    model_weight_path = "./weights/best.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
### 2. 测试图像数据增强
    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(int(image_size * 1.143)),
                                   transforms.CenterCrop(image_size)])
### 3. 开始预测
    predict = []
    class_list = []
    for filename in file_list:
        print(filename)
        filepath = os.path.join(root, filename)
        # 读取nii
        nii_volume = nib.load(filepath)
        volume = nii_volume.get_fdata().squeeze()
        # 调整图像大小
        volume = resize_volume(volume, h=image_size, w=image_size, d=d)
        img = test_transform(volume).unsqueeze(0)
        with torch.no_grad():
            # 预测图像类别
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            class_list.append(class_indict[str(predict_cla)])
### 4. 保存提交文件
    # 生成提交结果的DataFrame，其中包括样本ID和预测类别。
    submit = pd.DataFrame(
        {
            'uuid': [int(x[:-4]) for x in file_list],   # 提取测试集文件名中的ID
            'label': class_list,                        # 预测的类别
        }
    )
    # 按照ID对结果排序并保存为CSV文件
    submit = submit.sort_values(by='uuid')
    submit.to_csv('submit.csv', index=None)
    print("Done!")
