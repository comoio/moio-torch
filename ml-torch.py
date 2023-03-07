import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoarder

class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 1].values
        self.length = len(df)


    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] ** 2, self.x[index]])
        y = torch.FloatTensor([self.y[index]])

        return x, y


    def __len__(self):
        return self.length

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Linear(2, 1)


    def forward(self, x):
        x = self.layer(x)

        return x

train_dataset = CustomDataset("./dataset.csv")
train_dataLoader = DataLoarder(train_dataset, batch_size = 128, shutffle = True, drop_last = True)

device = "cuda" if  torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.00001)


checkpoint = 1

for epoch in range(10000):
    cost = 0.0


    for x, y in train_dataLoader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optim.step()

        cost += loss

    cost = cost / len(train_dataset)

    if (epoch + 1) % 1000 == 0:
        torch.save(
                {
                    "model" : "CustomModel"
                    "epoch" : epoch,
                    "model_state_dict" : model.state_dict(),
                    "optimizer_satt_dict" : optimizer.state_dict(), 
                    "cost" : cost,
                    "description" : f"CustomModel 체크포인트-{checkpoint}",
                },
                f"./checkpoint-{checkpoint}.pt",
         )
        checkpoint += 1    
