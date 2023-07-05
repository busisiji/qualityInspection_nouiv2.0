import os 
import time 
import copy
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
num_epochs = 30
# 数据集存放路径
path = 'data'
# 默认使用cpu加速
print("is use gpu:",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
needInitDataSet = True
folders = os.listdir(os.path.join(path,"train"))
classes_num = len(folders)
# 遍历数据集
if(needInitDataSet):
    shutil.rmtree(os.path.join(path, "val"))
    for folder in folders:
        # 图片格式为.jpg或.png
        jpg_files = glob.glob(os.path.join(path,"train", folder, "*.bmp"))
        png_files = glob.glob(os.path.join(path,"train", folder, "*.png"))
        files = jpg_files + png_files

        # 统计训练集数据
        num_of_img = len(files)
        print("Total number of {} image is {}".format(folder, num_of_img))
        # 从训练集里面抽取20%作为验证集
        shuffle = np.random.permutation(num_of_img)
        percent = int(num_of_img * 0.2)
        print("Select {} img as valid image".format(percent) )
        # 新建val文件夹存放验证集数据
        path_val = os.path.join(path,"val",folder)
        os.makedirs(path_val)

        # 把训练集里面抽取20%的数据复制到val文件夹
        # shuffle()方法将序列的所有元素随机排序
        for i in shuffle[:percent]:
            print("copy file {} ing".format(files[i].split('\\')[-1]))
            shutil.copy(files[i], path_val)

# 构建数据转换列表
# Resize(size,interpolation=2): 重置图像分辨率
# Grayscale(num_output_channels=1): 将图片转换为灰度图
# RandomRotation(degrees): 依degrees随机旋转一定角度
# RandomVerticalFlip(0.5): 以0.5的概率垂直翻转给定的PIL图像
# RandomHorizontalFlip(0.5): 以0.5的概率水平翻转给定的PIL图像
# ToTensor(): 将PIL_Image或ndarray转换为tensor,并且归一化至[0-1],即直接除以255
# Normalize(mean,std): 对数据按通道进行标准化,即先减均值,再除以标准差,注意是hwc
data_transforms = {
    'train':transforms.Compose([
        transforms.Resize((120,120)),
        transforms.Grayscale(3),
        transforms.RandomRotation(5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(), 
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])        
    ]),
    'val':transforms.Compose([
        transforms.Resize((120,120)),
        transforms.Grayscale(3),
        transforms.RandomRotation(5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(), 
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])  
    ])
}
def write_file(data):
    # 将文本写入到text02.txt文件
    with open('lables.txt', 'w') as file_read:
        file_read.write(data)

data_dir = './data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val'] }
print(type(image_datasets["train"]))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'val'] }

dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
write_file(str(class_names))
print(class_names)
# 从训练集中拿出一批图像
# 用iter和next函数来获取取一个批次的图片数据和其对应的图片标签
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
#imshow(out, title=[class_names[x] for x in classes])

# 定义训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs -1))
        print('-' * 10)
        # 每轮训练都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 将模型设置为训练模式
            else:
                model.eval()   # 将模型设置为评估模式 
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 清零梯度参数
                optimizer.zero_grad()
                # 前向传播, 只在训练阶段跟踪梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # 只在训练阶段进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                # deep copy 模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()  
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model
# 展示训练样本图片
def imshow(inp, title = None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5,0.5,0.5])
    std = np.array([0.5,0.5,0.5])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)
# 可视化预测模型
def visualize_model(model, num_images = 6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                print('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])  
                if images_so_far == num_images:
                    model.train(mode= was_training)
                    return
        model.train(mode==was_training)
def mymain():
    # 加载预训练模型resnet18网络层
    model_ft = models.resnet18(pretrained=True)
    # torch.save(model_ft.state_dict(), frozen_dir.app_path() + r'/quality_testing_ft.pkl')
    # model_ft = torch.load(frozen_dir.app_path() + r'/quality_testing_ft.pkl')
    num_ftrs = model_ft.fc.in_features
    # 重写全连接层n种分类
    print(classes_num)
    model_ft.fc = nn.Linear(num_ftrs, classes_num)
    model_ft = model_ft.to(device)
    # 交叉熵,多分类后不用手动添加softmax层
    criterion = nn.CrossEntropyLoss()
    # 整体微调,lr表示学习率
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # 以0.1的比率每轮调整学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    # 开始训练模型,默认训练批次200次
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    # 可视化预测模型
    # visualize_model(model_ft)
    # 保存模型
    torch.save(model_ft.state_dict(), "./quality_testing.pkl")
#运行主函数
if __name__ == "__main__":
    mymain()


