import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# --- Setup Directories ---
output_dir = "realsense_validation_dataset"
depth_dir = os.path.join(output_dir, "depth_data")
rgb_dir = os.path.join(output_dir, "rgb_images")
for d in [depth_dir, rgb_dir]:
    if not os.path.exists(d): os.makedirs(d)

pipeline = rs.pipeline()
config = rs.config()
width, height, fps = 1280, 720, 30
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Buffers to hold data in RAM
rgb_buffer = []
depth_buffer = []

recording = False
print("Controls: 'r' to START recording | 'q' to STOP and COMPILE")

try:
    while True:
        frames = pipeline.wait_for_frames(10000)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # We use .copy() to ensure the data in the list doesn't change 
        # when the camera hardware updates the frame buffer
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()

        depth_norm = cv2.convertScaleAbs(depth_image, alpha=0.05) 

        # 2. Create an empty HSV image
        hsv_depth = np.zeros((height, width, 3), dtype=np.uint8)

        # 3. Set Hue, Saturation, and Value
        # We use depth_norm directly for Hue (0 is Red, 120 is Blue/Cyan)
        hsv_depth[:, :, 0] = depth_norm  
        hsv_depth[:, :, 1] = 255         
        hsv_depth[:, :, 2] = 255         

        # 4. Convert HSV back to BGR for OpenCV to display it
        depth_colormap = cv2.cvtColor(hsv_depth, cv2.COLOR_HSV2BGR)

        # 5. Mask out the "Zero" (invalid) depth pixels
        depth_colormap[depth_image == 0] = 0

        if recording:
            rgb_buffer.append(color_image)
            depth_buffer.append(depth_image)
            
            # Visual indicator for the RGB window
            cv2.circle(color_image, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(color_image, f"RECORDING: {len(rgb_buffer)}", (50, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(color_image, "LIVE PREVIEW - Press 'r' to Record", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- SEPARATE WINDOWS DISPLAY ---
        # Resizing just for the screen so they aren't massive, 
        # but the saved data stays 1280x720.
        cv2.imshow('RGB Stream (Alignment Reference)', cv2.resize(color_image, (854, 480)))
        cv2.imshow('Depth Stream (Metric Visualization)', cv2.resize(depth_colormap, (854, 480)))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and not recording:
            recording = True
            print("--- RECORDING TO RAM ---")
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    # --- Compilation Phase ---
    if len(rgb_buffer) > 0:
        print(f"\nCaptured {len(rgb_buffer)} frames. Compiling video and saving data...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(f'{output_dir}/validation_video.mp4', fourcc, fps, (width, height))

        for i in range(len(rgb_buffer)):
            # 1. Save to Video
            out_video.write(rgb_buffer[i])
            # 2. Save PNG (for Depth Anything V2)
            cv2.imwrite(os.path.join(rgb_dir, f"frame_{i:05d}.png"), rgb_buffer[i])
            # 3. Save Metric Depth (NPY)
            np.save(os.path.join(depth_dir, f"depth_{i:05d}.npy"), depth_buffer[i])
            
            if i % 10 == 0:
                print(f"Processing frame {i}/{len(rgb_buffer)}...", end='\r')

        out_video.release()
        print("\nCompilation Complete! Data is ready in the folder.")
    else:
        print("No frames were recorded.")