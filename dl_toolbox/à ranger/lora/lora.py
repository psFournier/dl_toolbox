import torchvision.transforms.v2 as v2
import dl_toolbox.datasets as datasets
from torch.utils.data import Subset, RandomSampler
import torch
from dl_toolbox.utils import CustomCollate
import schedulefree
from functools import partial
import torch.nn.functional as F 
import timm
import torch.nn as nn
import minlora 

transform = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

NB_IMG = 45*700
dataset = datasets.Resisc('/data/NWPU-RESISC45', transform, 'all45')
trainset = Subset(dataset, indices=[i for i in range(NB_IMG) if 100<=i%700])
valset = Subset(dataset, indices=[i for i in range(NB_IMG) if 100>i%700])

train_loader = torch.utils.data.DataLoader(
    trainset,
    collate_fn=CustomCollate(),
    num_workers=6,
    pin_memory=True,
    sampler=RandomSampler(
        trainset,
        replacement=True,
        num_samples=5000
    ),
    drop_last=True,
    batch_size=4,
)
val_loader = torch.utils.data.DataLoader(
    valset,
    collate_fn=CustomCollate(),
    num_workers=6,
    pin_memory=True,
    shuffle=False,
    drop_last=True,
    batch_size=8,
)

def train(model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    optimizer.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch['image'], batch['label']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, criterion, optimizer, device, test_loader):
    model.eval()
    optimizer.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch['image'], batch['label']
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_lora_config(rank):
    return {  # specify which layers to add lora to, by default only add to linear layers
        nn.Linear: {
            "weight": partial(minlora.LoRAParametrization.from_linear, rank=rank),
        },
    }

class vit_ft(nn.Module):
    def __init__(self, freeze, lora, rank):
        super().__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True, global_pool='token')
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if lora:
            cfg = get_lora_config(rank)
            minlora.add_lora(self.encoder, lora_config=cfg)
        self.head = nn.Linear(self.encoder.num_features, 45)
            
    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = x[:, self.encoder.num_prefix_tokens:].mean(dim=1)
        x = self.head(x)
        return x
    
def name_is_lora(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["lora_A", "lora_B"]
    )

def name_is_head(name):
    return (name.split(".")[0]) == "head"

def lora_or_head(name):
    return name_is_lora(name) or name_is_head(name)

def get_params_by_name(model, print_shapes=False, name_filter=None):
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            if print_shapes:
                print(n, p.shape)
            yield p

torch.manual_seed(1)

model = vit_ft(freeze=True, lora=True, rank=4)
print([p[0] for p in list(model.named_parameters())][-10:])
parameters = list(model.parameters())
trainable_parameters = list(get_params_by_name(model, name_filter=lora_or_head))
print(
    f"The model will start training with only {sum([int(torch.numel(p)) for p in trainable_parameters])} "
    f"trainable parameters out of {sum([int(torch.numel(p)) for p in parameters])}."
)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.AdamW(
#    trainable_parameters,
#    lr=1e-3,
#)
optimizer = schedulefree.AdamWScheduleFree(trainable_parameters, lr=0.0025)

device = 'cuda'
model = model.to(device)
for epoch in range(1, 10):
    train(model, criterion, device, train_loader, optimizer, epoch)
    test(model, criterion, optimizer, device, val_loader)