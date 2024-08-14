from torchvision.models import efficientnet_v2_l
from attacks.boundary_attack import BoundaryAttack
import torch
from utils import show_img, load_image_480


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('using gpu') if torch.cuda.is_available() else print('using cpu')

init_img_path = 'picture/original/doge.png'
target_img_path = 'picture/original/grumpy_cat.png'

init_img = load_image_480(init_img_path)
target_img = load_image_480(target_img_path)

model = efficientnet_v2_l(pretrained=True)
attack = BoundaryAttack(model, device)
adv_img, mean_squared_error, all_calls = (
    attack(init_img, target_img=target_img, targeted=True, max_steps=100))

print('mean squared error: ', mean_squared_error)
print('all calls: ', all_calls)

show_img(adv_img, model)
