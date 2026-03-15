"""
I3D Feature Extractor
Extracts I3D features from video files
Source: https://github.com/v-iashin/video_features/models/i3d/extract_i3d.py
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision
from typing import Dict
from pathlib import Path
from collections import defaultdict

# Handle relative imports
_module_dir = Path(__file__).parent.parent.parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from video_features.models.i3d.i3d_src.i3d_net import I3D
from video_features.models.raft.raft_src.raft import RAFT, InputPadder
from video_features.transforms import (
    Clamp, PermuteAndUnsqueeze, PILToTensor, ResizeImproved, 
    ScaleTo1_1, TensorCenterCrop, ToFloat, ToUInt8
)
from video_features.utils import (
    reencode_video_with_diff_fps, form_slices, 
    dp_state_to_normal, show_predictions_on_dataset
)


class ExtractI3D:
    """I3D Feature Extractor for video understanding"""
    
    def __init__(self, args=None):
        """
        Initialize I3D extractor
        
        Args:
            args: Configuration object with attributes:
                - streams: 'rgb' or 'flow' or ['rgb', 'flow']
                - flow_type: 'raft' (currently only option)
                - stack_size: Number of frames to stack (default 64)
                - step_size: Step size for sliding window (default 64)
                - extraction_fps: Target fps for extraction (None for original)
                - device: torch device
                - output_path: Path for saving features (optional)
        """
        if args is None:
            args = type('Args', (), {})()
            
        # Default configuration
        self.streams = getattr(args, 'streams', ['rgb'])
        if isinstance(self.streams, str):
            self.streams = [self.streams]
        if self.streams is None or (isinstance(self.streams, list) and len(self.streams) == 0):
            self.streams = ['rgb']  # Default to RGB only for simplicity
            
        self.flow_type = getattr(args, 'flow_type', 'raft')
        self.stack_size = getattr(args, 'stack_size', 64)
        self.step_size = getattr(args, 'step_size', 64)
        self.extraction_fps = getattr(args, 'extraction_fps', None)
        self.device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.output_path = getattr(args, 'output_path', None)
        self.tmp_path = getattr(args, 'tmp_path', './tmp')
        self.keep_tmp_files = getattr(args, 'keep_tmp_files', False)
        
        # Dimensions for I3D
        self.min_side_size = 256
        self.central_crop_size = 224
        self.i3d_classes_num = 400
        
        # Initialize models
        self.models = self._load_models()
        
        # Transforms
        self.resize_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            ResizeImproved(self.min_side_size),
            PILToTensor(),
            ToFloat(),
        ])
        
        self.i3d_transforms = {
            'rgb': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ]),
            'flow': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                Clamp(-20, 20),
                ToUInt8(),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ])
        }
    
    def _load_models(self) -> Dict:
        """Load pre-trained I3D models"""
        models = {}
        
        # Determine checkpoint paths
        script_dir = Path(__file__).parent.parent.parent.parent  # vad_unified root
        checkpoints_dir = script_dir / 'video_features' / 'models' / 'i3d' / 'checkpoints'
        
        i3d_weights_paths = {
            'rgb': str(checkpoints_dir / 'i3d_rgb.pt'),
            'flow': str(checkpoints_dir / 'i3d_flow.pt'),
        }
        
        # Try to download checkpoints if they don't exist
        for stream, path in i3d_weights_paths.items():
            if not os.path.exists(path):
                print(f"Warning: I3D {stream} checkpoint not found at {path}")
                print(f"You need to download the checkpoints from: https://github.com/v-iashin/video_features")
                print(f"Or use --download_checkpoints flag")
        
        # Load I3D models for each stream
        for stream in self.streams:
            try:
                i3d_model = I3D(num_classes=self.i3d_classes_num, modality=stream)
                
                if os.path.exists(i3d_weights_paths[stream]):
                    state_dict = torch.load(i3d_weights_paths[stream], map_location='cpu')
                    state_dict = dp_state_to_normal(state_dict)
                    i3d_model.load_state_dict(state_dict)
                    print(f"Loaded I3D {stream} model from {i3d_weights_paths[stream]}")
                else:
                    print(f"Warning: Using uninitialized I3D {stream} model")
                
                i3d_model = i3d_model.to(self.device)
                i3d_model.eval()
                models[stream] = i3d_model
            except Exception as e:
                print(f"Error loading I3D {stream} model: {e}")
                raise
        
        return models
    
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """
        Extract I3D features from a video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with stream names as keys and feature arrays as values
        """
        # Re-encode video if needed
        original_video_path = video_path
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(
                video_path, self.tmp_path, self.extraction_fps
            )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract frames and features
        feats_dict = {stream: [] for stream in self.streams}
        timestamps_ms = []
        rgb_stack = []
        stack_counter = 0
        padder = None
        
        print(f"Extracting I3D features from {original_video_path}")
        
        first_frame = True
        while cap.isOpened():
            frame_exists, rgb = cap.read()
            
            if first_frame:
                first_frame = False
                if frame_exists is False:
                    continue
            
            if frame_exists:
                # Preprocess frame
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = self.resize_transforms(rgb)
                rgb = rgb.unsqueeze(0)
                
                if 'flow' in self.streams and padder is None:
                    padder = InputPadder(rgb.shape)
                
                rgb_stack.append(rgb)
                
                # Process when stack is full
                if len(rgb_stack) - 1 == self.stack_size:
                    batch_feats = self._run_on_stack(rgb_stack, stack_counter, padder)
                    for stream in self.streams:
                        feats_dict[stream].extend(batch_feats[stream].cpu().numpy().tolist())
                    
                    rgb_stack = rgb_stack[self.step_size:]
                    stack_counter += 1
                    timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            else:
                cap.release()
                break
        
        # Clean up
        if (self.extraction_fps is not None) and (not self.keep_tmp_files):
            try:
                os.remove(video_path)
            except:
                pass
        
        # Convert lists to numpy arrays
        output = {}
        for stream in self.streams:
            output[stream] = np.array(feats_dict[stream], dtype=np.float32)
        
        output['fps'] = np.array(fps, dtype=np.float32)
        output['timestamps_ms'] = np.array(timestamps_ms, dtype=np.float32)
        
        print(f"Extracted {len(output[self.streams[0]])} feature vectors of shape {output[self.streams[0]][0].shape if len(output[self.streams[0]]) > 0 else 'empty'}")
        
        return output
    
    def _run_on_stack(self, rgb_stack, stack_counter, padder=None) -> Dict[str, torch.Tensor]:
        """
        Process a stack of frames and extract features
        
        Shape progression:
        - rgb_stack is list of [1, C, 256, 256] frames (after resize_transforms)
        - After torch.cat: [T, C, 256, 256] where T = stack_size + 1
        - After [:-1]: [T-1, C, 256, 256] (drop last frame for optical flow compatibility)
        - After i3d_transforms: [1, C, T-1, 224, 224] (PermuteAndUnsqueeze handles permutation)
        - Model input: [1, 3, 64, 224, 224] for RGB
        """
        # Concatenate frames: list of [1, C, 256, 256] → [T, C, 256, 256]
        rgb_stacked = torch.cat(rgb_stack).to(self.device)
        
        batch_feats = {}
        for stream in self.streams:
            if stream == 'rgb':
                # Drop last frame to match optical flow frame count (65 frames → 64 frames)
                stream_slice = rgb_stacked[:-1]  # [T-1, 3, 256, 256]
            elif stream == 'flow':
                # For flow: extract optical flow from consecutive frame pairs
                # PLACEHOLDER: Currently using RGB. In production, use RAFT to extract flow
                stream_slice = rgb_stacked[:-1]  # [T-1, 3, 256, 256]
            else:
                raise NotImplementedError(f"Stream {stream} not implemented")
            
            # Apply stream-specific transforms
            # Input: [T-1, 3, 256, 256] (treating T-1 as batch dimension)
            # TensorCenterCrop: [T-1, 3, 256, 256] → [T-1, 3, 224, 224]
            # ScaleTo1_1: [T-1, 3, 224, 224] → [T-1, 3, 224, 224] (normalize to [-1, 1])
            # PermuteAndUnsqueeze: [T-1, 3, 224, 224] → [1, 3, T-1, 224, 224]
            stream_slice = self.i3d_transforms[stream](stream_slice)
            
            # After PermuteAndUnsqueeze, stream_slice is already [1, C, T-1, H, W]
            # No further dimension manipulation needed
            assert stream_slice.dim() == 5, f"Expected 5D tensor after transforms, got {stream_slice.dim()}D with shape {stream_slice.shape}"
            
            with torch.no_grad():
                features = self.models[stream](stream_slice, features=True)  # (1, 1024)
            
            batch_feats[stream] = features
        
        return batch_feats
