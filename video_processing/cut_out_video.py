import cv2

input = "D:/2024-04-18/box2/video.avi" #1920x1200 input file
output = "D:/2024-04-18/box2/cutout_video2.avi" #512x512 output file
square_size = (512,512) # size of the cutout 
x_start = 430
y_start = square_size[1] - 256 # 256 approximated case by case

def get_cutout(input, output, square_size=(512, 512), x_start=0, y_start=0):
    cap = cv2.VideoCapture(input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 90, square_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #top left corner to crop
        x = x_start # 1920/4 
        y = (frame.shape[0]- y_start)
        cropped_frame = frame[y:y+square_size[1], x:x+square_size[0]]
        out.write(cropped_frame)
    
    cap.release()
    out.release()

get_cutout(input, output, square_size, x_start, y_start)