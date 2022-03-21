# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch

import argparse
from PIL import ImageFile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import smdebug.pytorch as smd
import logging
import os
import json
import time
import sys
from torchvision import datasets
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO: Import dependencies for Debugging and Profiling


def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss = 0
    running_corrects = 0

    with torch.no_grad():  # We do not want to caluculate gradients while testing
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * inputs.size(0)  # calculate the running loss
            # calculate the running corrects
            running_corrects += pred.eq(labels.view_as(pred)).sum().item()

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects / len(test_loader.dataset)
        logger.info("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(
                test_loader.dataset), 100.0 * total_acc
        ))


def train(model, train_loader, validation_loader, criterion, optimizer, epoch, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook = get_hook(create_if_not_exists=True)
    epoch = 3
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0
        # TODO: Finish the rest of the training code
        # The code should stop training when the validation accuracy
        # stops increasing
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                if running_samples % 2000 == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        running_samples,
                        len(image_dataset[phase].dataset),
                        100.0 * (running_samples /
                                 len(image_dataset[phase].dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                    )
                    )

                # NOTE: Comment lines below to train and test on whole dataset
                if running_samples > (0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

        if loss_counter == 1:
            break
    return model


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(
        pretrained=True)  # using a pretrained resnet50 model

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 256),  # Adding our own fully connected layers
                             nn.ReLU(inplace=True),
                             nn.Linear(256, 133),
                             # output with 133 nodes - 133 classes of dog breeds
                             nn.ReLU(inplace=True)
                             )
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    batch_size = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(
        root=test_data_path, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size)

    validation_data = torchvision.datasets.ImageFolder(
        root=validation_data_path, transform=test_transform)
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=batch_size)

    return train_data_loader, test_data_loader, validation_data_loader


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = Net()
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--model", type=str, default="resnet50")

    opt = parser.parse_args()

    for key, value in vars(opt).items():
        print(f"{key}:{value}")
    # create model
    net = models.__dict__[opt.model](pretrained=True)
    if opt.gpu == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net.to(device)

    # Start the training.
    median_time = train(opt, net, device)
    print("Median training time per Epoch=%.1f sec" % median_time)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    # TODO: Add your optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train(model, train_data_loader, validation_loader,
          loss_criterion, optimizer, epoch, device, hook)

    model = train(model, train_data_loader, loss_criterion, optimizer)

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_data_loader, loss_criterion, device, hook)
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    save_model(model, args.model_dir)
    logger.info("Completed Saving the Model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
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
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list,
                        default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str,
                        default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str,
                        default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str,
                        default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int,
                        default=os.environ["SM_NUM_GPUS"])
    parser.add_argument('--output_dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()

    main(args)
