import torch 
import torch.nn as nn 
from torch.autograd import Variable 
from torch import optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time 

batch_size = 32 
num_epoches = 50
num_workers = 4 
learning_rate = 1e-2
use_gpu = torch.cuda.is_available()

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

class NerualNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(NerualNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out 

model = NerualNetwork(28*28, 300, 100, 10)
if use_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epoches):
    print("epoch {}".format(epoch + 1))
    print("*" * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data 
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = out.max(1)
        num_correct = (pred==label).sum()
        running_acc += num_correct.data[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print("[{}/{}] Loss: {:.6f}, Acc: {:.6f}".format(epoch, num_epoches, running_loss / (batch_size * i), running_acc / (batch_size * i)))
    print("Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}".format(epoch, running_loss / len(train_dataset), running_acc / len(train_dataset)))
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for data in test_loader:
        img, label = data 
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Varialbe(img, volatile=True)
            label = Varialbe(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = out.max(1)
        num_correct = (pred==label).sum()
        eval_acc += num_correct.data[0]
    print("Test Loss: {:.6f}, Acc: {:.6f}".format(eval_loss / len(test_dataset), eval_acc / len(test_dataset)))
    print()

torch.save(model.state_dict(), './nerual_network.pth')





