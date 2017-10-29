import torch
from torch import nn, optim
import torch.nn.functional as F 
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets 
from logger import Logger 

batch_size = 128 
learning_rate = 1e-3 
num_epoches = 20 

def to_np(x):
    return x.cpu().data.numpy()

train_dataset = datasets.MNIST(root='./data', transform=transforms.ToTensor(), download=True, train=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5, stride=1, padding=0), 
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(400, 120), 
            nn.Linear(120, 84), 
            nn.Linear(84, n_class))
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 

model = Cnn(1, 10)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
logger = Logger('./cnnlogs')

for epoch in range(num_epoches):
    print("epoch {}".format(epoch + 1))
    print("*" * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data 
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # forward propagation
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred==label).sum()
        accuracy = (pred==label).float().mean()
        running_acc += num_correct.data[0]
        # backward propagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step = epoch * len(train_loader) + i 
        # (1) Log the scalar values
        info = {'loss': loss.data[0], 'accuracy': accuracy.data[0]}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)
        # (3) Log the images 
        info = {'images': to_np(img.view(-1, 28, 28)[:10])}
        for tag, images in info.items():
            logger.image_summary(tag, images, step)

        if i % 300 == 0:
            print("[{}/{}] Loss: {:.6f}, Acc: {:.6f}".format(epoch, num_epoches, running_loss / (batch_size * i), running_acc / (batch_size * i)))
    print("Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}".format(epoch, running_loss / len(train_dataset), running_acc / len(train_dataset)))
    # evaluate mode for dropout or batchnorm
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for data in test_loader:
        img, label = data 
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1) 
        num_correct = (pred==label).sum()
        eval_acc += num_correct.data[0] 
    print("Test Loss: {:.6f}, Acc: {:.6f}".format(eval_loss / len(test_dataset), eval_acc / len(test_dataset)))
# save the parameters
torch.save(model.state_dict(), './cnn.pth')

