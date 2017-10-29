import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import time

batch_size = 64
learning_rate = 1e-3
num_epoches = 100
use_gpu = torch.cuda.is_available()
workers = 8

train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transforms.ToTensor(),
                              download=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
class LogisticRegression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegression, self).__init__()
        self.logistic = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.logistic(x)
        return out 

model = LogisticRegression(28*28, 10)
if use_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch+1))
    print('*'*10)
    start = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader):
        image, label = data 
        image = image.view(image.size(0), -1)
        if use_gpu:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        else:
            image = Variable(image)
            label = Variable(label)
        out = model(image)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred==label).sum()
        running_acc += num_correct.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print("[{}/{}] Loss: {:.6f}, Acc: {:.6f}".format(epoch+1, num_epoches, running_loss / (batch_size * (i + 1)), running_acc / (batch_size * (i + 1))))
    print("Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}".format(epoch + 1, running_loss / len(train_dataset), running_acc / len(train_dataset)))

    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for data in test_loader:
        image, label = data 
        image = image.view(image.size(0), -1)
        if use_gpu:
            image = Variable(image, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            image = Variable(image, volatile=True)
            label = Variable(label, volatile=True)
        out = model(image)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred==label).sum()
        eval_acc += num_correct.data[0]
    print("Test Loss: {:.6f}, Acc: {:.6f}".format(eval_loss / len(test_dataset), eval_acc / len(test_dataset)))
    print("Time: {:.1f} s".format(time.time() - start))
    print()

torch.save(model.state_dict(), './logistic.pth')


        





