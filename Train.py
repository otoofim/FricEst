import torchvision.models as models
from torchvision import transforms
from frictionDataloader import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torchvision
import wandb
import os
import warnings
from statistics import mean
from UNet import *
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat.float(),y.float()) + self.eps)
        return loss



def train(**kwargs):


    kwargs["model_add"] = "./checkpoints/{}_{}/best.pth".format(kwargs["project_name"], kwargs["run_name"])
    hyperparameter_defaults = {
        "run": kwargs["run_name"],
        "hyper_params": kwargs,
    }

    base_add = os.getcwd()


    if kwargs['continue_tra']:
        wandb.init(config = hyperparameter_defaults, project = kwargs["project_name"], entity = kwargs["entity"],
                    name = hyperparameter_defaults['run'], resume = "must", id = kwargs["wandb_id"])
        print("wandb resumed...")
    else:
        wandb.init(config = hyperparameter_defaults, project = kwargs["project_name"], entity = kwargs["entity"],
                    name = hyperparameter_defaults['run'], resume = "allow")


    val_every = 5
    img_w = kwargs["input_img_dim"][0]
    img_h = kwargs["input_img_dim"][1]


    dataset_arg = {
            "season": kwargs["season"],
            "date": kwargs["date"],
            "hour": kwargs["hour"],
            "year": kwargs["year"]
    }

    preprocess_in = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        transforms.Resize((img_w,img_h))
    ])


    dataset = FrictionLoader(datasetConfig = dataset_arg, transform_in = preprocess_in)
    train_split, val_split = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    train_loader = DataLoader(dataset = train_split, batch_size = kwargs["batch_size"], shuffle = True, drop_last = True)
    val_loader = DataLoader(dataset = val_split, batch_size = kwargs["batch_size"], shuffle = True, drop_last = True)


    if kwargs['device'] == "cpu":
        device = torch.device("cpu")
        print("Running on the CPU")
    elif kwargs['device'] == "gpu":
        device = torch.device(kwargs['device_name'])
        print("Running on the GPU")



    model = UNet(img_w)
    
    if kwargs["continue_tra"]:
        model.load_state_dict(torch.load(kwargs["model_add"])['model_state_dict'])
        print("model state dict loaded...")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr =  kwargs["learning_rate"], weight_decay = kwargs["momentum"])
    criterion = RMSELoss()


    if kwargs["continue_tra"]:
        optimizer.load_state_dict(torch.load(kwargs["model_add"])['optimizer_state_dict'])
        print("optimizer state dict loaded...")


    tr_loss = {"loss":0.}
    val_loss = {"loss":0.}
    best_val = 1e10
    wandb.watch(model)

    start_epoch = 0
    resume_position = 0
    end_epoch = kwargs["epochs"]
    total = kwargs["epochs"]

    if kwargs["continue_tra"]:
        start_epoch = torch.load(kwargs["model_add"])['epoch'] + 1
        resume_position = start_epoch

    with tqdm(range(start_epoch, end_epoch), initial = resume_position, total = total, unit="epoch", leave = True, position = 0) as epobar:
        for epoch in epobar:
                
                epobar.set_description("Epoch {}".format(epoch + 1))
                epobar.set_postfix(ordered_dict = {'tr_loss':tr_loss["loss"], 'val_loss':val_loss["loss"]})
                tr_loss = {"loss":0.}


                with tqdm(train_loader, unit="batch", leave = False) as batbar:
                    for i, batch in enumerate(batbar):
                        
                        batbar.set_description("Batch {}".format(i + 1))
                        optimizer.zero_grad()
                        model.train()
                        
                        #forward pass
                        estMU = model(batch['image'].to(device))
                        loss = criterion(batch['label'].to(device), estMU.squeeze())
                        loss.backward()
                        optimizer.step()
                        tr_loss["loss"] += loss.item()
                    tr_loss["loss"] /= len(train_loader)
                wandb.log({"loss_train": tr_loss["loss"], "epoch": epoch + 1})

                          
                if ((epoch+1) % val_every == 0):
                    with tqdm(val_loader, unit="batch", leave = False) as valbar:
                        with torch.no_grad():
                            val_loss = {"loss":0.}
                            
                            for i, batch in enumerate(valbar):
                                                                             
                                valbar.set_description("Val_batch {}".format(i + 1))
                                model.eval()
                                optimizer.zero_grad()
                                loss_sum = 0.                                                       
                             
                                estMU = model(batch['image'].to(device))
                                loss = criterion(batch['label'].to(device), estMU.squeeze())
                                val_loss["loss"] += loss.item()
                                    

                            val_loss["loss"] /= len(val_loader)

                                                                                         
                        wandb.log({"loss_val": val_loss["loss"], "epoch": epoch + 1})



                        if val_loss["loss"] < best_val:

                            newpath = os.path.join(base_add, "checkpoints", hyperparameter_defaults['run'])

                            if not os.path.exists(os.path.join(base_add, "checkpoints")):
                                os.makedirs(os.path.join(base_add, "checkpoints"))

                            if not os.path.exists(newpath):
                                os.makedirs(newpath)

                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'tr_loss': tr_loss,
                                'val_loss': val_loss,
                                'hyper_params': kwargs,
                                }, os.path.join(newpath, "best.pth"))
