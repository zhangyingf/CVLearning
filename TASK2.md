# CV实践打卡笔记二  

## 增加数据集交叉验证，训练多个模型

### 通过不同的随机数种子实现不同的验证集、训练集划分结果（TODO: 调整训练集验证集比例）
        seedlist = [0,1,121,311]
        for seed in seedlist:
            train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root=args.data_path, val_rate=0.2, seed=seed)

## 加载多个模型，投票出最终结果
### 将多个模型结果累加决策出最终结果(TODO: 调整投票策略)
        # load model weights
        seedlist = [0,1,121,311]
        for seed in seedlist:
            model_weight_path = "./weights/seed{}/best.pth".format(seed)
            model.load_state_dict(torch.load(model_weight_path, map_location=device))
            model.eval()
            # predict class
            with torch.no_grad():
                if output is None:
                    output = torch.squeeze(model(img.to(device))).cpu()
                else:
                    output += torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        class_list.append(class_indict[str(predict_cla)])