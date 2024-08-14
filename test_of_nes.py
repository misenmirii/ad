from torchvision.models import efficientnet_v2_l
from attacks.NES import NES
import torch
from utils import show_img, load_image_480

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('using gpu') if torch.cuda.is_available() else print('using cpu')

# 图像加载和预处理
image_path = './picture/img.png'  # where picture is

# 准备输入图像
input_image = load_image_480(image_path)
label = torch.tensor([66])  # 目标类（可以是标签或目标）

model = efficientnet_v2_l(pretrained=True)
# attack = BoundaryAttack(model, mean=effi_mean, size=effi_size, device=device)
# attack('./picture/original/doge.png', './picture/original/grumpy_cat.png', targeted=True)
attack = NES(model, device)
adv_img = attack(input_image, label, targeted=True, step=100, epsilon=0.08)

show_img(adv_img, model)
