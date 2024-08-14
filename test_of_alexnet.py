import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model.alexnet import alexnet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using gpu') if torch.cuda.is_available() else print('using cpu')

# 加载预训练的模型
mod = alexnet.alexnet(pretrained=True)  # here to change model
mod = mod.to(device)
mod.eval()

# 图像加载和预处理
image_path = './picture/img.png'  # where picture is
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

output = mod(image)

# 获取模型预测标签
_, predicted = torch.max(output, 1)
original_label = predicted.item()
probs = torch.softmax(output, dim=1)
top_probs, top_labels = torch.topk(probs, k=1)
final_pro = top_probs[0]

# 加载ImageNet类别标签文件
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# 寻找标签名
search_word_no = classes[original_label]
line_with_search_word_no = None

with open('imagenet_synsets.txt', 'r') as file:
    for line in file:
        if search_word_no in line:
            line_with_search_word_no = line
            break

# 获取类别名称
original_class_name = line_with_search_word_no

# 显示标签和类别名称
plt.figure(figsize=(16, 9))
plt.subplot(1, 1, 1)
# 将CUDA设备上的张量移动到CPU
image_cpu = image.squeeze(0).cpu().detach().permute(1, 2, 0)
plt.imshow(image_cpu)
plt.title(f'Original Image\nPredicted Label: {original_label}\nPredicted Class: {original_class_name}\nPrediction Confidence: {final_pro.item():.2f}')
plt.axis('off')

plt.show()
