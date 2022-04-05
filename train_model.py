import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import argparse
import smdebug.pytorch as smd

# Some images fail to load...
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_loader, criterion, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    print(f"Accuracy: {100 * total_acc}%, Testing Loss: {total_loss}")


def train(model, train_loader, criterion, optimizer, hook):
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    trained_images = 0
    num_images = len(train_loader.dataset)
    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        trained_images += len(inputs)
        loss.backward()
        optimizer.step()
        print(f"{trained_images}/{num_images} images trained...")


def net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133))

    return model


def create_data_loaders(data_train, data_test, batch_size):
    train_transforms = transforms.Compose([
        # transforms.RandomRotation(30),
        # transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()])
    
    trainset = datasets.ImageFolder(root=data_train, transform=train_transforms)
    testset = datasets.ImageFolder(root=data_test, transform=test_transforms)
    
    return (
        torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    args = parser.parse_args()

    model = net()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(loss_criterion)


    train_loader, test_loader = create_data_loaders(args.train, args.test, args.batch_size)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_criterion, optimizer, hook)
        test(model, test_loader, loss_criterion, hook)

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    main()


