import cv2
import os

def video_to_frames(video_path, output_folder, desired_fps):
    """
    Extracts frames from a video at a specified fps and saves them as images in the specified folder.

    :param video_path: Path to the video file.
    :param output_folder: Path to the folder where frames will be saved.
    :param desired_fps: Desired frames per second for extraction.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get original video fps
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / desired_fps)
    
    if frame_interval <= 0:
        print("Error: Desired FPS is higher than the video's FPS.")
        return
    
    frame_number = 0
    saved_frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break
        
        # Save frames only at the specified interval
        if frame_number % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_number:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            saved_frame_number += 1
        
        frame_number += 1
    
    # Release the video capture object
    cap.release()
    print(f"Video has been successfully split into frames at {desired_fps} FPS.")

# Example usage:
video_path = r"C:\Users\zakhar.statkevich\Downloads\telloCV-master\telloCV-master\Seq1_camera1.mov"
output_folder = "photos"
desired_fps = 2  # Extract 5 frames per second
video_to_frames(video_path, output_folder, desired_fps)
