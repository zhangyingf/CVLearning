import os
import argparse

import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import BrainPETSet as MyDataSet

from ADCCPANet import  MT
from p2twindowdwattn import p2t_tiny


from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    #tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

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

    # model = MT(image_size=img_size, in_channels=d, num_classes=2).to(device)
    model = p2t_tiny(img_size=img_size, in_chans=d, num_classes=2).to(device)

    file_name = "PET"
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    #pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

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

        torch.save(model.state_dict(), "./weights/{}.pth".format(epoch))
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

    output2 = "epoch:{},best_acc:{:.4f},best_f1:{:.4f}".format(best_epoch, best_acc, best_f1)
    with open(file_name, "a+") as f:
        f.write(output2 + '\n')
        f.close


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./TrainData")

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
