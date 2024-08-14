from timm.utils import reparameterize_model

from fastervit import create_model
import torch
from fastervit import create_model
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('using gpu') if torch.cuda.is_available() else print('using cpu')

model = create_model('faster_vit_0_224',
                     pretrained=True,
                     model_path="./fastervit_3_224_1k.pth.tar")
# Extract the actual state dictionary
# To Train from scratch/fine-tuning

# checkpoint = torch.load('./model/FastViT/fastvit_sa12_reparam.pth.tar')


# print("Checkpoint keys:")
# for key in checkpoint.keys():
#      print(key)
#


# model_keys = set(model.state_dict().keys())
# checkpoint_keys = set(checkpoint['state_dict'].keys())
#
# missing_keys = model_keys - checkpoint_keys
# unexpected_keys = checkpoint_keys - model_keys
#
# print("\nMissing keys in checkpoint:")
# for key in missing_keys:
#     print(key)
#
# print("\nUnexpected keys in checkpoint:")
# for key in unexpected_keys:
#     print(key)

# 加载新的 state_dict
# model.load_state_dict(checkpoint['new_state_dict'], strict=False)


# # ... train ...
# model.load_state_dict(torch.load('./model/FastViT/fastvit_sa12_reparam.pth.tar'))
# Load unfused pre-trained checkpoint for fine-tuning
# # or for downstream task training like detection/segmentation
# # checkpoint = torch.load('./model/FastViT/fastvit_sa12_reparam.pth.tar')
# # model.load_state_dict(checkpoint['state_dict'])

# 查看模型state_dict中的所有键
# model.load_state_dict(torch.load('./model/FastViT/fastvit_sa12_reparam.pth.tar'))
# ... train ...

# For inference
model.to(device).eval()
model_inf = reparameterize_model(model).eval()
# Use model_inf at test-time


# 图像加载和预处理
image_path = './picture/dog.jpg'  # where picture is
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()

])
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

image = image.float()

output = model_inf(image)

# 打印每个键对应权重的统计信息
# for key, value in state_dict.items():
#     print(f"\nKey: {key}")
#     print(f"Shape: {value.shape}")
#     print(f"Min: {value.min().item()}, Max: {value.max().item()}, Mean: {value.mean().item()}")
#
# # 检查一个特定层的权重值（例如 patch_embed.0.rbr_conv.0.conv.weight）
# key_to_check = "patch_embed.0.rbr_conv.0.conv.weight"
# if key_to_check in state_dict:
#     weight = state_dict[key_to_check]
#     print(f"\nValues of {key_to_check}:")
#     print(weight)
# else:
#     print(f"\nKey {key_to_check} not found in state_dict.")


# 获取模型预测标签
_, predicted = torch.max(output, 1)
original_label = predicted.item()
probs = torch.softmax(output, dim=1)
top_probs, top_labels = torch.topk(probs, k=1)
final_pro = top_probs[0]

probabilities = torch.softmax(output, dim=1)

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
plt.title(
    f'Original Image\nPredicted Label: {original_label}\nPredicted Class: {original_class_name}\nPrediction Confidence: {final_pro.item():.2f}')
plt.axis('off')

plt.show()
