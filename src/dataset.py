import os
from PIL import Image
from torch.utils.data import Dataset
import torch

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent

class VQADataset(Dataset):
    def __init__(self, data, label2idx,
                 img_feature_extractor, text_tokenizer,
                 device, transform=None, img_dir="data\\vqa_coco_dataset\\val2014-resised"):
        self.data = data
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.img_feature_extractor = img_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(base_dir, self.img_dir, self.data[index]["image_path"])
        img = Image.open(img_path).convert("RGB")

        if (self.transform):
            img = self.transform(img)

        if (self.img_feature_extractor):
            img = self.img_feature_extractor(images=img, return_tensors="pt")
            img = {k: v.to(self.device).squeeze(0) 
                   for k, v in img.items()}

        question = self.data[index]["question"]
        if (self.text_tokenizer):
            question = self.text_tokenizer(
                question,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_tensors="pt"
            )
            question = {k: v.to(self.device).squeeze(0)
                        for k, v in question.items()}

        label = self.data[index]["answer"]
        label = torch.tensor(
            self.label2idx[label], dtype=torch.long
        ).to(self.device)

        sample = {
            "image": img,
            "question": question,
            "label": label
        }

        return sample