from model.T2T_ViT.T2T_ViT.models.t2t_vit import *
from model.T2T_ViT.T2T_ViT.utils import load_for_transfer_learning
import torch
from timm.models import create_model
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('using gpu') if torch.cuda.is_available() else print('using cpu')

# create model
model = t2t_vit_24()

# load the pretrained weights
# change num_classes based on dataset, can work for different image size as we interpolate the position embeding for different image size.

load_for_transfer_learning(model, 'model/T2T_ViT/82.3_T2T_ViT_24.pth.tar', use_ema=True, strict=False, num_classes=1000)


# For inference
model.eval()
# Use model_inf at test-time
model.to(device)


# 图像加载和预处理
image_path = './picture/img.png'  # where picture is
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)


# 应用softmax函数，将输出转化为概率分布
probabilities = torch.softmax(output, dim=1)
# 获取模型预测标签
_, predicted = torch.max(output, 1)
original_label = predicted.item()
probs = torch.softmax(output, dim=1)
top_probs, top_labels = torch.topk(probs, k=1)
final_pro = top_probs[0]



# 获取预测概率
predicted_prob = probabilities[0][original_label].item()

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

