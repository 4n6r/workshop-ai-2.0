import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)
  def forward(self, x):
     out = self.l1(x)
     out = self.relu(out)
     out = self.l2(out)
     return out
  
def train(model, device, train_loader, optimizer, epoch, log_interval):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        images = data.reshape(-1,28*28)
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            images = data.reshape(-1,28*28)
            images, target = images.to(device), target.to(device)
            output = model(images)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

def main():
  # batch size for training
  batch_size = 64
  # batch size for testing
  test_batch_size = 1000
  # define input size
  input_size = 784
  # define hidden size
  hidden_size = 250
  # define epochs
  num_epoch = 10
  # define learning rate
  learning_rate = 0.001
  log_interval= 10

  # define device on which the model should run
  device = (
      "cuda"
      if torch.cuda.is_available()
      else "mps"
      if torch.backends.mps.is_available()
      else "cpu"
  )
  print(f"Using {device} device")

  # get training and test dataset from mnist
  train_dataset = torchvision.datasets.MNIST(root="./data", train=True, 
                                             transform=transforms.ToTensor(), download=True)
  test_dataset = torchvision.datasets.MNIST(root="./data", train=False, 
                                            transform=transforms.ToTensor(), download=False)
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

  img, label = train_dataset[5]
  print(img)
  print(label)

  model = NeuralNetwork(input_size, hidden_size, 10).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(1, num_epoch + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)


if __name__ == "__main__":
    main()