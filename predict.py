
from typing import Tuple, Dict

from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image


class Predicter():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = mobilenet_v3_large(weights=True)
        # self.model.classifier[3] = nn.Linear(in_features=1280, out_features=25, bias=True)
        self.model.classifier[3] = nn.Linear(in_features=1280, out_features=80, bias=True)
        self.model.load_state_dict(
            # torch.load('train_checkpoint/MobileNetv3_large_epoch_28.pth', map_location=self.device)["model_state_dict"])
            torch.load('checkpoint/MobileNetv3_large_epoch_64.pth', map_location=self.device)["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.set_seed()

    def set_seed(self, seed=66):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def img_load(self, img_path: str) -> Tensor:
        """

        Args:
            img_path: 图片的路径

        Returns:
            x: 返回图片的张量格式
        """
        # TODO: 看下opencv奇奇怪怪的问题
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert("RGB")
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img_tensor = transformer(img)
        x = img_tensor.unsqueeze(0).to(self.device)
        return x

    def scene_classify(self,
                       img_path: str,
                       threshold: float = 0.6) -> Dict:
        """

        Args:
            img_path: 传入的图片路径参数
            threshold: 阈值，默认值为0.6

        Returns:
            {
                label_index: 预测的标签的索引
                score: 预测的得分
            }
        """
        x = self.img_load(img_path=img_path)
        output = self.model(x)
        probability = F.softmax(output, dim=1)
        prob, index = torch.max(probability, 1)

        score, label_index = prob.item(), index.item()
        if prob < threshold:
            label_index = 25

        result_dict = {label_index: score}

        return result_dict


if __name__ == '__main__':
    img = Image.open('test_iamge/b1d0a191-06deb55d.jpg')
    scene_predicter = Predicter()
    result = scene_predicter.scene_classify("test_iamge/b1d0a191-06deb55d.jpg")
    print(result)

    # TODO: 修改下读取同片的方式，png格式是32位读取，而jpg是24位深度的
