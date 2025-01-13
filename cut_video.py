import cv2
video_length = 60 # in seconds
input_file = "" # video file to divide
output_dir = "" # output directory to save the files.

def split_video(path, length, output):

    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_clip = length*fps
    current_clip = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if len(frames) == frames_per_clip:
            out = cv2.VideoWriter(f'{output}_{current_clip}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            for f in frames:
                out.write(f)

            out.release()
            frames = []
            current_clip += 1
            print(f"Video: {current_clip} is processed")

    cap.release()


split_video(input_file, video_length, output_dir)