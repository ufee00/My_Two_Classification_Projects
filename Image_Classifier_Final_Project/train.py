# Imports here
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

data_dir='/home/workspace/ImageClassifier/flowers'

def load_data(data_dir):
    # Defining transforms for the training, validation, and testing sets
    # Training set transformations: random rotation, random resized crop, random horizontal flip, normalization
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(255),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Validation and test set transformations: resize, center crop, normalization
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder and apply the respective transformations
    train_datasets = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_datasets = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_datasets = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)

    return trainloader, validloader, testloader

def build_model():
    # Load pre-trained VGG16 model
    model = models.vgg16(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(4096, 1024)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(1024, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    return model

def train_model(model, trainloader, validloader, epochs, device):
    # Define the optimizer, Learning rate scheduler and the criterion
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    criterion = nn.NLLLoss()

    model.to(device)
    
    for epoch in range(epochs):
        train_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                valid_loss += batch_loss.item()

                # Calculating accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Adjusting learning rate
        scheduler.step()

        # Printing statistics
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader)*100:.2f}%")
        
    return model

def save_checkpoint(model, train_datasets, filename='checkpoint.pth'):
    model.class_to_idx = train_datasets.class_to_idx
    model.cpu()
    torch.save({'arch': 'vgg16',
                'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                filename)

def main():
    parser = argparse.ArgumentParser(description='Train a neural network on flower images')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing flower images')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the checkpoint')
    
    args = parser.parse_args()
    
    trainloader, validloader, testloader = load_data(args.data_dir)
    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_model(model, trainloader, validloader, args.epochs, device)
    save_checkpoint(model, trainloader.dataset, args.save_dir)

if __name__ == "__main__":
    main()