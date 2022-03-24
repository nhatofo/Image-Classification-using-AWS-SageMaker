#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
import logging
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader):
    model.eval()
    running_corrects=0
    for (inputs, labels) in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
    total_acc = running_corrects / len(test_loader.dataset)
    logger.info(f"Test set: Average accuracy: {100*total_acc}")
    

def train(model, train_loader, epochs, criterion, optimizer):
    # https://github.com/awslabs/sagemaker-debugger/blob/master/docs/pytorch.md
    model.train()
    count = 0
    for e in range(epochs):
        print(e)
        for (inputs, labels) in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += len(inputs)
            if count > 400:
                break
    return model

def net():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model

def create_data_loaders(data, batch_size, test_batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    
    trainset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    validset = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    testset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    
    return (
        torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(validset, batch_size=test_batch_size, shuffle=False),
        torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False))

def main(args):
    model=net()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    train_loader, valid_loader, test_loader = create_data_loaders(args.data, args.batch_size, args.test_batch_size)
    
    model=train(model, train_loader, args.epochs, loss_criterion, optimizer)
    
    test(model, test_loader)
    
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument('--data', type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Momentum: {args.momentum}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Test Batch Size: {args.test_batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    
    main(args)

