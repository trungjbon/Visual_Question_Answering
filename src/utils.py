import random
import numpy as np
import torch

SEED = 59

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {SEED}")

set_seed(SEED)

def read_data(path):
    data = []

    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split("\t")
            qa = temp[1].split("?")

            if (len(qa) == 3):
                answer = qa[2].strip()
            else:
                answer = qa[1].strip()

            data_sample = {
                "image_path": temp[0][:-2],
                "question": qa[0] + "?",
                "answer": answer
            }
            data.append(data_sample)

    return data

def evaluate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    losses = []

    with (torch.no_grad()):
        for idx, inputs in enumerate(dataloader):
            images = inputs["image"]
            questions = inputs["question"]
            labels = inputs["label"]

            outputs = model(images, questions)

            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        loss = sum(losses) / len(losses)
        acc = correct / total

        return loss, acc
    
def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []

        model.train()

        for idx, inputs in enumerate(train_loader):
            images = inputs["image"]
            questions = inputs["question"]
            labels = inputs["label"]

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss, val_acc = evaluate(
            model, val_loader,
            criterion
        )
        val_losses.append(val_loss)

        print(f"EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\tVal Acc: {val_acc}")

        scheduler.step()

    return train_losses, val_losses