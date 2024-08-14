from torchvision.models import efficientnet_v2_l
from attacks.sign_opt import OPT_attack_sign_SGD
import torch
from utils import show_img, load_image_480


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('using gpu') if torch.cuda.is_available() else print('using cpu')

init_img_path = 'picture/img.png'

init_img = load_image_480(init_img_path)

model = efficientnet_v2_l(pretrained=True)
model.eval()
output = model(init_img)
_, predicted = torch.max(output, 1)
label = predicted.item()
print(label)
attack = OPT_attack_sign_SGD(model)
adv_img = attack(init_img, label)

show_img(adv_img, model)
