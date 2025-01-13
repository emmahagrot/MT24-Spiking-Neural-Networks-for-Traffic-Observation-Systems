import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

cuda_device = 1
start_video_nr = 0
end_video_nr = 10 
decay_rate = 1 
frame_width = 640
frame_height = 480

decay = False # set true to add decay to the input events

torch.cuda.set_device(device=cuda_device)

for video in range(start_video_nr, end_video_nr):
    video_nr = video
    
    # Read event and timestamp data
    df_events = pd.read_csv(f'clip_{video_nr + 1}.csv') #+1 is just a temporary solution for wrongly named .csv files
    df_timestamps = pd.read_csv('timestamps.csv', skiprows=range(0,(video_nr*27000)), nrows=27000) 

    # Extracting data from the events DataFrame
    x_coords = df_events.iloc[:, 0].to_numpy().tolist()  # x-coordinates
    y_coords = df_events.iloc[:, 1].to_numpy().tolist()  # y-coordinates
    polarities = df_events.iloc[:, 2].to_numpy().tolist()  # polarities 
    
    # Normalize so each series starts at 0 to match the clip and the timestamps
    timestamps1 = df_events.iloc[:, 3].to_numpy() 
    timestamps = (timestamps1 - timestamps1[0]).tolist()
    time_windows1 = df_timestamps.iloc[:, 1].to_numpy()
    time_windows = time_windows1 - time_windows1[0]

    # Function to process and aggregate events into frames
    def aggregate_events_spatially(x_coords, y_coords, timestamps, time_windows, width, height):
        frames = []
        event_index = 0
        n_events = len(timestamps)  

        # Iterate through each time window
        for i in tqdm(range(1, len(time_windows)-1)):
            start_time = time_windows[i - 1] 
            end_time = time_windows[i + 1] 

           
            frame = np.zeros((height, width))

            # Process events falling within the current time window
            while event_index < n_events and timestamps[event_index] < end_time:
                if timestamps[event_index] >= start_time:
                    event_x = x_coords[event_index]
                    event_y = y_coords[event_index]
                    polarity = polarities[event_index]
                    time_since_start = timestamps[event_index]- start_time
                    
                    if decay: # add decay rate if desired
                        decay_multiplier = np.exp(-time_since_start/decay_rate)  
                        frame[event_y, event_x] = polarity*decay_multiplier
                    
                    else:
                        frame[event_y, event_x] = polarity*decay_multiplier 

                event_index += 1

            frames.append(torch.tensor(frame).to_sparse())


        return torch.stack(frames)


    frames = aggregate_events_spatially(x_coords, y_coords, timestamps, time_windows, frame_width, frame_height)
    print("Processing complete.")

    # Review outputs
    filename = f"event_frames_{video_nr}.pt"
    print(frames.shape)
    torch.save(frames, filename)
    print(f"Saved frames to:{filename}")

    print(frames.size)
    print(frames.shape)
    plt.subplot()
    x_min, y_min, x_max, y_max = (55, 224, 311, 480) #(120, 224, 312, 416) Hardcoded after what was a good cutout
    plt.imshow(frames[200].to_dense()[y_min:y_max, x_min:x_max],cmap='gray' )
    plt.axis('off')
    plt.savefig("test_frame.png", bbox_inches='tight', pad_inches=0)

