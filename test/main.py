import sys
sys.path.append("Project")
from src.utils import *
from src.dataset import *
from src.model import *
from torchvision import transforms
from transformers import ViTImageProcessor, AutoTokenizer
from torch.utils.data import DataLoader

# Load train data
train_set_path = "Project\\data\\vqa_coco_dataset\\vaq2.0.TrainImages.txt"
train_data = read_data(train_set_path)

# Load val data
val_set_path = "Project\\data\\vqa_coco_dataset\\vaq2.0.DevImages.txt"
val_data = read_data(val_set_path)

# Load test data
test_set_path = "Project\\data\\vqa_coco_dataset\\vaq2.0.TestImages.txt"
test_data = read_data(test_set_path)

classes = set([sample["answer"] for sample in train_data])
label2idx = {
    cls_name: idx for idx, cls_name in enumerate(classes)
}
idx2label = {
    idx: cls_name for idx, cls_name in enumerate(classes)
}

print(f"Classes: {idx2label}")

data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
        transforms.CenterCrop(size=180),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
])

img_feature_extractor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)
text_tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base"
)
device = "cuda" if (torch.cuda.is_available()) else "cpu"

train_dataset = VQADataset(
    train_data,
    label2idx=label2idx,
    img_feature_extractor=img_feature_extractor,
    text_tokenizer=text_tokenizer,
    device=device,
    transform=data_transform
)
val_dataset = VQADataset(
    val_data,
    label2idx=label2idx,
    img_feature_extractor=img_feature_extractor,
    text_tokenizer=text_tokenizer,
    device=device,
)
test_dataset = VQADataset(
    test_data,
    label2idx=label2idx,
    img_feature_extractor=img_feature_extractor,
    text_tokenizer=text_tokenizer,
    device=device,
)

train_batch_size = 256
test_batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=test_batch_size,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False
)

n_classes = len(classes)
hidden_size = 256
dropout_prob = 0.2

text_encoder = TextEncoder().to(device)
visual_encoder = VisualEncoder().to(device)
classifier = Classifier(
    hidden_size=hidden_size,
    dropout_prob=dropout_prob,
    n_classes=n_classes
).to(device)

model = VQAModel(
    visual_encoder=visual_encoder,
    text_encoder=text_encoder,
    classifier=classifier
).to(device)

model.freeze()

lr = 1e-3
epochs = 50

scheduler_step_size = epochs * 0.8
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=scheduler_step_size,
    gamma=0.1
)

train_losses, val_losses = fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs
)

val_loss, val_acc = evaluate(
    model,
    val_loader,
    criterion
)
test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion
)

print("Evaluation on val/test dataset")
print("Val accuracy: ", val_acc)
print("Test accuracy: ", test_acc)