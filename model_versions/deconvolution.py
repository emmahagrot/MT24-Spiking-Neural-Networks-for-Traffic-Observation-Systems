
import torch
import norse
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loading import get_data
import snntorch.functional as sf
from norse.torch import LICell, LILinearCell, LIParameters
from norse.torch.module.lif import LIFCell, LIFParameters
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc
import json
from datetime import date

# Set device
cuda_kernel = 1
today = date.today()
torch.cuda.set_device(cuda_kernel)
num_inputs = 200*200
num_outputs = 64*64 #4096
tau_list = [180]
decay_list = [1e-3, 1e-4]
lr_list = [1e-3, 1e-4]

# Loss function
loss_function = nn.MSELoss()


def save_data(model, train_loss, validation_loss, tau, today, epoch):
    file_name = f'{today}_Tau_{tau}_deconvLI'
    torch.save(model.state_dict(), f"{file_name}.pth")
    
    data = {
        'Epoch': epoch,
        'Tau': tau,
        'train_loss': train_loss,
        'validation_loss': validation_loss,
    }
    
    with open(f"{file_name}.json", 'w') as f:
        json.dump(data, f)
        

for tau_mem in tau_list:   
    print(tau_mem) 
    class SNN(nn.Module):
        def __init__(self):
            super(SNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=15, stride=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.lif1 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.conv2 = nn.Conv2d(8, 8, kernel_size=13, stride=1)
            self.bn2 = nn.BatchNorm2d(8)
            self.lif2 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.conv3 = nn.Conv2d(8, 8, kernel_size=11, stride=1)
            self.bn3 = nn.BatchNorm2d(8)
            self.lif3 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            
            self.conv4 = nn.Conv2d(8, 8, kernel_size=9, stride=1)
            self.bn4 = nn.BatchNorm2d(8)
            self.lif4 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            
            self.conv5 = nn.Conv2d(8, 8, kernel_size=7, stride=1)
            self.bn5 = nn.BatchNorm2d(8)
            self.lif5 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            
            self.conv6 = nn.Conv2d(8, 8, kernel_size=5, stride=1)
            self.bn6 = nn.BatchNorm2d(8)
            self.lif6 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            
            self.conv7 = nn.Conv2d(8, 8, kernel_size=3, stride=1)
            self.bn7 = nn.BatchNorm2d(8)
            self.lif7 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            
            self.deconv = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, output_padding=0)
            self.bn8 = nn.BatchNorm2d(4)
            self.lif8 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
            
            self.deconv2 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=3, padding=1, output_padding=0)
            self.bn9 = nn.BatchNorm2d(1)
            self.lif9 = LICell(p=LIFParameters(tau_mem_inv=tau_mem))
            
            self.maxpool = nn.MaxPool2d(2,2)
            self.dropout = nn.Dropout(p=0.5)


        def forward(self, x, mem_states):
            batch_size, C, W, H = x.shape
            x = (x != 0).float()

            mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8, mem9 = mem_states
            

            v1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(v1, mem1)

            v2 = self.dropout(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(v2, mem2)
            
            v3 = self.dropout(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(v3, mem3)
            
            v4 = self.dropout(self.bn4(self.conv4(spk3)))
            spk4, mem4 = self.lif4(v4, mem4)
            
            v5 = self.dropout(self.bn5(self.conv5(spk4)))
            spk5, mem5 = self.lif5(v5, mem5)
            
            v6 = self.dropout(self.bn6(self.conv6(spk5)))
            spk6, mem6 = self.lif6(v6, mem6)
            
            v7 = self.dropout(self.bn7(self.conv7(spk6)))
            spk7, mem7 = self.lif7(v7, mem7)
            
            deconv_output1 = self.dropout(self.bn8(self.deconv(spk3)))
            spk8, mem8 =self.lif8(deconv_output1, mem8)
            
            deconv_output2 = self.dropout(self.bn9(self.deconv2(deconv_output1)))
            spk8, mem9 = self.lif9(deconv_output2, mem9)
        
            return spk8.squeeze(1), (mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8, mem9)



    def loss_fn(output_frame, target_frame, step):
        mse_loss = loss_function(output_frame, target_frame*1000) # Multiplication due to the numbers being too small, should be fixed when creating the data

        return mse_loss

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNN().to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params}")
    trainable_weights = sum(p.numel() for name, p in model.named_parameters() 
                        if p.requires_grad and 'weight' in name)

    print(f"Total trainable weights: {trainable_weights}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    num_epochs = 15
    frame_size = (64, 64)
    sequence_length = 75
    train_loss_list = []
    val_loss_list = []
    overlap = 25

    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        num_train_batches = 0
        
        for data_nr in range(26):
            data = get_data(data_nr)
            
            if data is not None:
                train_data, val_data = data
                print(f"File {data_nr} has been loaded. Training mode.")
                train_loader  = train_data

                for i, (frames, targets) in enumerate(train_loader):
                    mem_states = (None, None, None, None, None, None, None, None, None)
                    
                    optimizer.zero_grad()

                    frames, targets = frames.to(device), targets.to(device)
                    
                    loss = 0
                    
                    for step in range(sequence_length):
                        input_frame = frames[:, step].unsqueeze(1)
                        output, mem_states = model(input_frame, mem_states)
                        
                        if step >= overlap:
                            final_output = output
                            loss += loss_fn(final_output, targets[:, step], step)/ (sequence_length-overlap) #only train on the last 50 frames
                        
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_train_batches += 1
        
        
        model.eval()
        val_loss = 0
        num_test_batches = 0
        with torch.no_grad():
            for data_nr in range(26):
                data = get_data(data_nr)
                
                if data is not None:
                    train_data, val_data = data
                    print(f"File {data_nr} has been loaded. Validation mode.")
                    validation_loader  = val_data
                    
                    for i, (frames, targets) in enumerate(validation_loader):
                        mem_states = (None, None, None, None, None)

                        frames, targets = frames.to(device), targets.to(device)
                        
                        loss = 0
                        accuracy = 0
                        
                        for step in range(sequence_length):
                            input_frame = frames[:, step].unsqueeze(1)
                            output,mem_states = model(input_frame, mem_states)
                            final_output = output
                            
                            loss += loss_fn(final_output, targets[:, step], step) /sequence_length
                        
                        val_loss += loss.item()
                        num_test_batches += 1
        
        
        del data
        gc.collect()
            
        print(f"Epoch {epoch+1} completed with train loss {train_loss/num_train_batches}, validation loss {val_loss/num_test_batches}")
        train_loss_list.append(np.round(train_loss/num_train_batches, 4))
        val_loss_list.append(np.round(val_loss/num_test_batches, 4))
    
    save_data(model, train_loss_list, val_loss_list, tau_mem, today, epoch)
