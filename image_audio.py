import cv2
import numpy as np
import os
import random
import subprocess
from pathlib import Path
import librosa

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    try:
        duration = librosa.get_duration(filename=str(audio_path))
        return duration
    except Exception as e:
        print(f"Could not read audio {audio_path}: {e}")
        return 3.0  # Default duration if audio can't be read

def get_max_dimensions(image_files):
    """Get the maximum width and height from all images"""
    max_width = 0
    max_height = 0
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            height, width = img.shape[:2]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
    
    return max_width, max_height

def pad_image_to_size(img, target_width, target_height):
    """Pad image to target size while maintaining aspect ratio and centering"""
    height, width = img.shape[:2]
    
    # Calculate padding
    pad_width = max(0, target_width - width)
    pad_height = max(0, target_height - height)
    
    # Center the image by splitting padding evenly
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    
    # Add padding (black borders)
    padded_img = cv2.copyMakeBorder(
        img, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]  # Black padding
    )
    
    return padded_img

def create_enhanced_random_transition(image_folder="images", audio_folder="audios", 
                                    output_file="final_video_with_audio.mp4",
                                    transition_duration=1.2, fps=30):
    """
    Create video with enhanced random transitions synced to audio duration
    Images keep their original size with black padding if needed
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
    
    print(f"Processing {len(image_files)} images with original sizes")
    
    # Find maximum dimensions across all images
    max_width, max_height = get_max_dimensions(image_files)
    print(f"Video dimensions will be: {max_width}x{max_height} (based on largest image)")
    
    # Get corresponding audio files and durations
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac']
    image_audio_pairs = []
    
    for img_path in image_files:
        img_stem = img_path.stem  # filename without extension
        audio_path = None
        
        # Look for matching audio file
        for ext in audio_extensions:
            potential_audio = Path(audio_folder) / f"{img_stem}{ext}"
            if potential_audio.exists():
                audio_path = potential_audio
                break
        
        if audio_path:
            duration = get_audio_duration(audio_path)
            image_audio_pairs.append((img_path, audio_path, duration))
            print(f"Image: {img_path.name} -> Audio: {audio_path.name} ({duration:.2f}s)")
        else:
            # No matching audio found, use default duration
            default_duration = 3.0
            image_audio_pairs.append((img_path, None, default_duration))
            print(f"Image: {img_path.name} -> No audio found, using {default_duration}s")
    
    if not image_audio_pairs:
        print("No valid image-audio pairs found!")
        return
    
    # Create temporary video file (without audio)
    temp_video = "temp_video_no_audio.mp4"
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (max_width, max_height))
    
    # Load images with original sizes (padded to max dimensions)
    images = []
    durations = []
    
    for img_path, audio_path, duration in image_audio_pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
        
        # Get original dimensions
        original_height, original_width = img.shape[:2]
        print(f"Original size of {img_path.name}: {original_width}x{original_height}")
        
        # Pad image to match video dimensions while keeping original size
        img_padded = pad_image_to_size(img, max_width, max_height)
        
        images.append(img_padded)
        durations.append(duration)
        print(f"Loaded and padded: {img_path.name}")
    
    if len(images) != len(durations):
        print("Mismatch between images and durations!")
        return
    
    transition_frames = int(transition_duration * fps)
    
    # Enhanced transition types
    transition_types = ['left', 'up', 'up_with_zoom']
    transitions = []
    
    for i in range(len(images) - 1):
        direction = random.choice(transition_types)
        transitions.append(direction)
        print(f"Transition {i+1}: {direction}")
    
    # Create video with audio-synced durations
    for i in range(len(images)):
        current_img = images[i]
        audio_duration = durations[i]
        image_frames = int(audio_duration * fps)
        
        print(f"Processing image {i+1}/{len(images)} - Duration: {audio_duration:.2f}s ({image_frames} frames)")
        
        # Show image for EXACT audio duration (no transition subtraction)
        for frame in range(image_frames):
            out.write(current_img)
        
        # Add transition frames AFTER the image duration
        if i < len(images) - 1:
            next_img = images[i + 1]
            transition_type = transitions[i]
            
            for frame in range(transition_frames):
                progress = frame / transition_frames
                eased_progress = 1 - (1 - progress) ** 3
                
                if transition_type == 'left':
                    frame_img = create_left_slide_frame(current_img, next_img, eased_progress, max_width, max_height)
                elif transition_type == 'up':
                    frame_img = create_up_slide_frame(current_img, next_img, eased_progress, max_width, max_height)
                elif transition_type == 'up_with_zoom':
                    frame_img = create_up_zoom_frame(current_img, next_img, eased_progress, max_width, max_height)
                else:
                    frame_img = create_left_slide_frame(current_img, next_img, eased_progress, max_width, max_height)
                
                out.write(frame_img)
        
        print(f"Completed image {i+1}/{len(images)}")
    
    out.release()
    
    # Calculate total video duration
    total_duration = sum(durations) + (len(images) - 1) * transition_duration
    print(f"Temporary video created: {temp_video}")
    print(f"Total video duration: {total_duration:.2f} seconds")
    print(f"Final video dimensions: {max_width}x{max_height}")
    
    # Create synced audio track
    audio_file = create_synced_audio_track(image_audio_pairs, transition_duration)
    
    # Combine video and audio using FFmpeg
    if audio_file:
        combine_video_audio_ffmpeg(temp_video, audio_file, output_file)
        
        # Clean up temporary files
        try:
            os.remove(temp_video)
            os.remove(audio_file)
            print("✅ Cleaned up temporary files")
        except:
            pass
    else:
        # No audio, just rename the video file
        os.rename(temp_video, output_file)
        print(f"✅ Video created without audio: {output_file}")

def create_synced_audio_track(image_audio_pairs, transition_duration):
    """Create audio track that matches the video timing"""
    try:
        import pydub
        from pydub import AudioSegment
        
        print("Creating synced audio track...")
        combined_audio = AudioSegment.empty()
        has_audio = False
        
        for i, (img_path, audio_path, duration) in enumerate(image_audio_pairs):
            if audio_path and audio_path.exists():
                # Load audio file
                audio = AudioSegment.from_file(str(audio_path))
                has_audio = True
                
                # Adjust audio to match expected duration
                expected_duration_ms = int(duration * 1000)
                if len(audio) > expected_duration_ms:
                    # Trim if audio is longer
                    audio = audio[:expected_duration_ms]
                elif len(audio) < expected_duration_ms:
                    # Pad with silence if audio is shorter
                    silence = AudioSegment.silent(duration=expected_duration_ms - len(audio))
                    audio = audio + silence
                
                combined_audio += audio
                print(f"Added audio: {audio_path.name} ({duration:.2f}s)")
                
                # Add transition silence (except for last audio)
                if i < len(image_audio_pairs) - 1:
                    transition_silence = AudioSegment.silent(duration=int(transition_duration * 1000))
                    combined_audio += transition_silence
                    print(f"Added transition silence: {transition_duration:.2f}s")
            else:
                # Create silence for images without audio
                silence_duration = int(duration * 1000)
                if i < len(image_audio_pairs) - 1:
                    silence_duration += int(transition_duration * 1000)
                silence = AudioSegment.silent(duration=silence_duration)
                combined_audio += silence
                print(f"Added silence for image: {img_path.name} ({duration:.2f}s)")
        
        if has_audio:
            # Export synced audio
            audio_output = "temp_synced_audio.wav"
            combined_audio.export(audio_output, format="wav")
            print(f"Synced audio track created: {audio_output}")
            return audio_output
        else:
            print("No audio files found")
            return None
        
    except ImportError:
        print("❌ pydub not installed. Install with: pip install pydub")
        print("Video will be created without audio.")
        return None
    except Exception as e:
        print(f"❌ Error creating audio track: {e}")
        return None

def combine_video_audio_ffmpeg(video_file, audio_file, output_file):
    """Combine video and audio using FFmpeg"""
    try:
        print(f"🎬 Combining video and audio using FFmpeg...")
        
        # FFmpeg command to combine video and audio
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', video_file,  # Input video
            '-i', audio_file,  # Input audio
            '-c:v', 'copy',    # Copy video stream (no re-encoding)
            '-c:a', 'aac',     # Encode audio as AAC
            '-strict', 'experimental',
            output_file
        ]
        
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Final video with audio created: {output_file}")
            print(f"🎉 Video duration matches audio perfectly!")
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            # Fallback: copy video without audio
            os.rename(video_file, output_file)
            print(f"⚠️ Created video without audio: {output_file}")
            
    except FileNotFoundError:
        print("❌ FFmpeg not found! Please install FFmpeg:")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   Mac: brew install ffmpeg")
        print("   Linux: sudo apt install ffmpeg")
        print("⚠️ Creating video without audio...")
        os.rename(video_file, output_file)
    except Exception as e:
        print(f"❌ Error combining video and audio: {e}")
        os.rename(video_file, output_file)

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
    # Check if folders exist
    if not os.path.exists("images"):
        print("❌ Images folder not found! Please create 'images' folder and add your photos.")
    elif not os.path.exists("audios"):
        print("❌ Audios folder not found! Please create 'audios' folder and add your audio files.")
    else:        
        print("🎬 Creating audio-synced enhanced random transition video with original image sizes...")
        create_enhanced_random_transition(
            image_folder="images",
            audio_folder="audios",
            output_file="final_video_with_audio.mp4",
            transition_duration=1.2,
            fps=30
        )
        
        print("✅ Video creation completed!")