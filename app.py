import cv2
import numpy as np
import os
import random
from pathlib import Path

def create_enhanced_random_transition(image_folder="images", output_file="enhanced_random.mp4",
                                    image_duration=3, transition_duration=1.2,
                                    fps=30, width=1920, height=1080):
    """
    Create video with enhanced random transitions including diagonal slides
    """
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = set()
    
    for ext in image_extensions:
        image_files.update(Path(image_folder).glob(f'*{ext}'))
        image_files.update(Path(image_folder).glob(f'*{ext.upper()}'))
    
    image_files = sorted(list(image_files))
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"Processing {len(image_files)} images with enhanced random transitions")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Load images
    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (width, height))
        images.append(img)
        print(f"Loaded: {img_path.name}")
    
    image_frames = int(image_duration * fps)
    transition_frames = int(transition_duration * fps)
    
    # Enhanced transition types
    transition_types = ['left', 'up', 'diagonal_up_left', 'left_with_fade', 'up_with_zoom']
    transitions = []
    
    for i in range(len(images) - 1):
        direction = random.choice(transition_types)
        transitions.append(direction)
        print(f"Transition {i+1}: {direction}")
    
    for i in range(len(images)):
        current_img = images[i]
        
        # Static phase
        if i < len(images) - 1:
            static_frames = image_frames - transition_frames
        else:
            static_frames = image_frames
            
        for frame in range(static_frames):
            out.write(current_img)
        
        # Transition phase
        if i < len(images) - 1:
            next_img = images[i + 1]
            transition_type = transitions[i]
            
            for frame in range(transition_frames):
                progress = frame / transition_frames
                eased_progress = 1 - (1 - progress) ** 3
                
                if transition_type == 'left':
                    frame_img = create_left_slide_frame(current_img, next_img, eased_progress, width, height)
                elif transition_type == 'up':
                    frame_img = create_up_slide_frame(current_img, next_img, eased_progress, width, height)
                # elif transition_type == 'diagonal_up_left':
                #     frame_img = create_diagonal_slide_frame(current_img, next_img, eased_progress, width, height)
                # elif transition_type == 'left_with_fade':
                #     frame_img = create_left_fade_frame(current_img, next_img, eased_progress, width, height)
                elif transition_type == 'up_with_zoom':
                    frame_img = create_up_zoom_frame(current_img, next_img, eased_progress, width, height)
                else:
                    frame_img = create_left_slide_frame(current_img, next_img, eased_progress, width, height)
                
                out.write(frame_img)
        
        print(f"Completed image {i+1}/{len(images)}")
    
    out.release()
    print(f"Enhanced random transition video created: {output_file}")

def create_left_slide_frame(current_img, next_img, progress, width, height):
    """Create left slide transition frame"""
    slide_pixels = int(width * progress)
    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Current image sliding left
    if slide_pixels < width:
        remaining_width = width - slide_pixels
        frame_img[:, :remaining_width] = current_img[:, slide_pixels:]
    
    # Next image sliding in from right
    if slide_pixels > 0:
        frame_img[:, width-slide_pixels:] = next_img[:, :slide_pixels]
    
    return frame_img

def create_up_slide_frame(current_img, next_img, progress, width, height):
    """Create up slide transition frame"""
    slide_pixels = int(height * progress)
    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Current image sliding up
    if slide_pixels < height:
        remaining_height = height - slide_pixels
        frame_img[:remaining_height, :] = current_img[slide_pixels:, :]
    
    # Next image sliding in from bottom
    if slide_pixels > 0:
        frame_img[height-slide_pixels:, :] = next_img[:slide_pixels, :]
    
    return frame_img

def create_diagonal_slide_frame(current_img, next_img, progress, width, height):
    """Create diagonal up-left slide transition"""
    slide_x = int(width * progress)
    slide_y = int(height * progress)
    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Current image sliding diagonally up-left
    if slide_x < width and slide_y < height:
        remaining_width = width - slide_x
        remaining_height = height - slide_y
        frame_img[:remaining_height, :remaining_width] = current_img[slide_y:, slide_x:]
    
    # Next image sliding in from bottom-right
    if slide_x > 0 and slide_y > 0:
        frame_img[height-slide_y:, width-slide_x:] = next_img[:slide_y, :slide_x]
    
    return frame_img

def create_left_fade_frame(current_img, next_img, progress, width, height):
    """Create left slide with fade effect"""
    slide_pixels = int(width * progress)
    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Current image with fade and slide
    alpha = 1.0 - progress * 0.7
    current_faded = (current_img * alpha).astype(np.uint8)
    
    if slide_pixels < width:
        remaining_width = width - slide_pixels
        frame_img[:, :remaining_width] = current_faded[:, slide_pixels:]
    
    # Next image sliding in with fade in
    alpha_next = 0.3 + progress * 0.7
    next_faded = (next_img * alpha_next).astype(np.uint8)
    
    if slide_pixels > 0:
        frame_img[:, width-slide_pixels:] = next_faded[:, :slide_pixels]
    
    return frame_img

def create_up_zoom_frame(current_img, next_img, progress, width, height):
    """Create up slide with zoom effect"""
    slide_pixels = int(height * progress)
    frame_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Current image with zoom out
    zoom_factor = 1.0 + progress * 0.3
    if zoom_factor > 1.0:
        zoomed_height = int(height * zoom_factor)
        zoomed_width = int(width * zoom_factor)
        current_zoomed = cv2.resize(current_img, (zoomed_width, zoomed_height))
        
        y_offset = (zoomed_height - height) // 2
        x_offset = (zoomed_width - width) // 2
        current_cropped = current_zoomed[y_offset:y_offset+height, x_offset:x_offset+width]
    else:
        current_cropped = current_img
    
    # Position current image (sliding up)
    if slide_pixels < height:
        remaining_height = height - slide_pixels
        frame_img[:remaining_height, :] = current_cropped[slide_pixels:, :]
    
    # Next image sliding in from bottom
    if slide_pixels > 0:
        frame_img[height-slide_pixels:, :] = next_img[:slide_pixels, :]
    
    return frame_img

if __name__ == "__main__":
    # Check if images folder exists
    if not os.path.exists("images"):
        print("Images folder not found! Please create 'images' folder and add your photos.")
    else:        
        print("Creating enhanced random transition video...")
        create_enhanced_random_transition(
            image_folder="images",
            output_file="enhanced_random.mp4",
            image_duration=3,
            transition_duration=1.2,
            fps=30,
            width=1920,
            height=1080
        )
        
        print("Video created successfully!")