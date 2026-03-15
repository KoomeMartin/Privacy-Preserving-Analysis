"""
Utilities for video feature extraction
"""

import os
import tempfile
import shutil
import subprocess
import torch
from pathlib import Path


def reencode_video_with_diff_fps(video_path: str, tmp_path: str, target_fps: int) -> str:
    """
    Re-encode video to target fps using ffmpeg.
    
    Args:
        video_path: Path to input video
        tmp_path: Path to temporary directory
        target_fps: Target frames per second
        
    Returns:
        Path to re-encoded video
    """
    os.makedirs(tmp_path, exist_ok=True)
    tmp_video_name = Path(video_path).stem + f'_{target_fps}fps.mp4'
    tmp_video_path = os.path.join(tmp_path, tmp_video_name)
    
    cmd = [
        'ffmpeg',
        '-loglevel', 'quiet',
        '-i', video_path,
        '-vf', f'fps={target_fps}',
        '-y',  # overwrite
        tmp_video_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"FFmpeg not found or failed to re-encode video: {e}")
    
    return tmp_video_path


def form_slices(n_frames: int, stack_size: int = 64, step_size: int = 64):
    """
    Form window slices for frame stacking.
    
    Args:
        n_frames: Total number of frames
        stack_size: Window size
        step_size: Step size between windows
        
    Yields:
        Tuples of (start_idx, end_idx)
    """
    start = 0
    while True:
        end = start + stack_size
        if end > n_frames:
            break
        yield start, end
        start += step_size


def dp_state_to_normal(state_dict):
    """Convert DataParallel state dict to normal state dict."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def show_predictions_on_dataset(logits, dataset_name='kinetics'):
    """Print top predictions (placeholder)."""
    pass
