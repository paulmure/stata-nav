import torch
import pickle
import torch.optim as optim

from models import *
from load_data import *

NUM_EPOCHS = 20


def test_model(net, val_loader):
    correct_pred = 0
    total_pred = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred += 1
                total_pred += 1

    accuracy = 100 * float(correct_pred) / total_pred
    print(f'Test accuracy is {accuracy:.1f} %')
    return accuracy


def train(net, optimizer, criterion, train_loader, val_loader, epochs):
    training_losses = []
    test_accs = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'{epoch}: loss: {running_loss}')
        training_losses.append(running_loss)
        test_accs.append(test_model(net, val_loader))

    print('Finished Training')
    return training_losses, test_accs


def train_baseline():
    num_classes, train_loader, val_loader = get_data_loader()
    baseline_model = Baseline(num_classes)
    optimizer = optim.Adam(baseline_model.parameters())
    criterion = nn.CrossEntropyLoss()
    hist = train(baseline_model, optimizer, criterion,
                 train_loader, val_loader, NUM_EPOCHS)
    with open('results/baseline.hist', 'wb') as fp:
        pickle.dump(hist, fp)
    torch.save(baseline_model, 'results/baseline.model')


def train_last_layer():
    num_classes, train_loader, val_loader = get_data_loader()
    model = ReplaceLastLayer(num_classes)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    hist = train(model, optimizer, criterion,
                 train_loader, val_loader, NUM_EPOCHS)
    with open('results/resnet_last_layer.hist', 'wb') as fp:
        pickle.dump(hist, fp)
    torch.save(model, 'results/resnet_last_layer.model')


def train_all():
    num_classes, train_loader, val_loader = get_data_loader()
    model = ReplaceAll(num_classes)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    hist = train(model, optimizer, criterion,
                 train_loader, val_loader, NUM_EPOCHS)
    with open('results/resnet_all.hist', 'wb') as fp:
        pickle.dump(hist, fp)
    torch.save(model, 'results/resnet_all.model')


def train_random_rotation():
    transform = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    num_classes, train_loader, val_loader = get_augmented_data_loader(
        transform)
    baseline_model = ReplaceAll(num_classes)
    optimizer = optim.Adam(baseline_model.parameters())
    criterion = nn.CrossEntropyLoss()
    hist = train(baseline_model, optimizer, criterion,
                 train_loader, val_loader, NUM_EPOCHS)
    with open('results/random_rotation.hist', 'wb') as fp:
        pickle.dump(hist, fp)
    torch.save(baseline_model, 'results/random_rotation.model')


def train_random_crop():
    transform = transforms.Compose([
        transforms.RandomCrop(1500),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    num_classes, train_loader, val_loader = get_augmented_data_loader(
        transform)
    baseline_model = ReplaceAll(num_classes)
    optimizer = optim.Adam(baseline_model.parameters())
    criterion = nn.CrossEntropyLoss()
    hist = train(baseline_model, optimizer, criterion,
                 train_loader, val_loader, NUM_EPOCHS)
    with open('results/random_crop.hist', 'wb') as fp:
        pickle.dump(hist, fp)
    torch.save(baseline_model, 'results/random_crop.model')


def train_random_blur():
    transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(7, 13), sigma=(6, 7)),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    num_classes, train_loader, val_loader = get_augmented_data_loader(
        transform)
    baseline_model = ReplaceAll(num_classes)
    optimizer = optim.Adam(baseline_model.parameters())
    criterion = nn.CrossEntropyLoss()
    hist = train(baseline_model, optimizer, criterion,
                 train_loader, val_loader, NUM_EPOCHS)
    with open('results/random_blur.hist', 'wb') as fp:
        pickle.dump(hist, fp)
    torch.save(baseline_model, 'results/random_blur.model')


train_baseline()
train_last_layer()
train_all()
train_random_rotation()
train_random_crop()
train_random_blur()
