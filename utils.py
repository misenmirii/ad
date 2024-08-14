import torch
import os
import shutil
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image


def distance(x_adv, x, norm='l2'):
    diff = (x_adv - x).view(x.size(0), -1)
    if norm == 'l2':
        out = torch.sqrt(torch.sum(diff * diff)).item()
        return out
    elif norm == 'linf':
        out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
        return out


def pick_gpu_lowest_memory():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_file = os.path.join(path, os.path.basename(script))
            shutil.copyfile(script, dst_file)


def show_img(adv_img, model):
    output = model(adv_img)
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
    image_cpu = adv_img.squeeze(0).cpu().detach().permute(1, 2, 0)
    plt.imshow(image_cpu)
    plt.title(
        f'Original Image\nPredicted Label: {original_label}\nPredicted Class: {original_class_name}\nPrediction Confidence: {final_pro.item():.2f}')
    plt.axis('off')

    plt.show()


def load_image_480(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((480, 480)),  # 根据模型输入大小调整
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 增加批次维度


def load_image_224(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据模型输入大小调整
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 增加批次维度
