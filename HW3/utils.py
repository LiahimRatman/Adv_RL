import torch
from torchvision import transforms


mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 256, 256 * 2)
t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 256, 256 * 2)
resize = transforms.Resize(250)


def init_normal_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def show_results(imgs_real, imgs_fake):
    for img_real, img_fake in zip(imgs_real, imgs_fake):
        img = torch.cat((img_real, img_fake), -1)
        img = img * t_std + t_mean
        img = resize(transforms.ToPILImage()(img).convert('RGB'))
        display(img)


def set_requires_grad(net, req_grad):
    for param in net.parameters():
        param.requires_grad = req_grad
