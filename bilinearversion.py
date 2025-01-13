
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loading import get_data
from norse.torch import LICell
from norse.torch.module.lif import LIFCell, LIFParameters
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc
import json
from datetime import date

# Set device
cuda_kernel = 0
today = date.today()
torch.cuda.set_device(cuda_kernel)
num_inputs = 200*200
num_outputs = 64*64 #4096
tau_list = [180]
decay_list = [1e-3, 1e-4]
lr_list = [1e-4]
loss_function = nn.MSELoss()

def save_data(model, train_loss, validation_loss, tau, today, epoch):
    file_name = f'{today}_Tau_{tau}_bilinearLI_upsample'
    
    torch.save(model.state_dict(), f"{file_name}.pth")
    
    data = {
        'Epoch': epoch,
        'Tau': tau,
        'train_loss': train_loss,
        'validation_loss': validation_loss,
    }
    
    with open(f"{file_name}.json", 'w') as f:
        json.dump(data, f)
        

for learning_rate in lr_list:   
    tau_mem = 180
    print(tau_mem) 
    class SNN(nn.Module):
        def __init__(self):
            super(SNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2)
            self.bn1 = nn.BatchNorm2d(8)
            self.lif1 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(8)
            self.lif2 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

            self.conv3 = nn.Conv2d(8, 1, kernel_size=1, stride=1)
            self.bn3 = nn.BatchNorm2d(1)
            self.lif3 = LICell(p=LIFParameters(tau_mem_inv=tau_mem))
            
            self.maxpool = nn.MaxPool2d(2,2)
            self.dropout = nn.Dropout(p=0.5)
            self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)
            self.upsample2 = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True)
            

        def forward(self, x, mem_states):
            batch_size, C, W, H = x.shape
            x = (x != 0).float()

            mem1, mem2, mem3, mem4= mem_states
            

            v1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(v1, mem1)

            v2 = self.dropout(self.bn2(self.conv2(self.maxpool(spk1))))
            spk2, mem2 = self.lif2(v2, mem2)
            
            v3 = self.dropout(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(v3, mem3)
            
            output1 = self.upsample(spk3)
            output = self.upsample2(output1)
            
            return output.squeeze(1), (mem1, mem2, mem3, mem4)



    def loss_fn(output_frame, target_frame, step):
        mse_loss = loss_function(output_frame, target_frame*1000) # Multiplication due to the numbers being too small, should be fixed when creating the data

        return mse_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params}")
    trainable_weights = sum(p.numel() for name, p in model.named_parameters() 
                        if p.requires_grad and 'weight' in name)

    print(f"Total trainable weights: {trainable_weights}")
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
                    mem_states = (None, None, None, None)
                    
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
                        mem_states = (None, None, None, None)

                        frames, targets = frames.to(device), targets.to(device)
                        loss = 0
                        
                        for step in range(sequence_length):
                            input_frame = frames[:, step].unsqueeze(1)
                            output, mem_states = model(input_frame, mem_states)
                            
                            #res, current, spikes = model(frames.unsqueeze(2))
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
    
