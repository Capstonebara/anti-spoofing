import os
import sys
import torch
import numpy as np
import cv2

# Add Silent-Face-Anti-Spoofing directory to path
base_dir = os.path.join(os.getcwd(), 'Silent-Face-Anti-Spoofing')
sys.path.append(base_dir)

# Import anti-spoofing modules
from src.generate_patches import CropImage
from src.utility import parse_model_name, get_kernel
from src.model_lib.MiniFASNet import MiniFASNetV1SE
from src.data_io import transform as trans

class AntiSpoofingPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.base_dir = base_dir
        
        # Define model path
        self.model_dir = os.path.join(base_dir, "resources", "anti_spoof_models")
        self.model_path = os.path.join(self.model_dir, "4_0_0_80x80_MiniFASNetV1SE.pth")
        
        # Image cropper utility
        self.image_cropper = CropImage()
        
        # Load test transforms
        self.test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        
        print(f"Anti-spoofing using device: {self.device}")
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            print(f"ERROR: Anti-spoofing model not found at: {self.model_path}")
    
    def preprocess_face(self, frame, face_bbox):
        """Preprocess detected face for anti-spoofing
        
        Args:
            frame: Original BGR frame
            face_bbox: Face bounding box [x, y, width, height]
            
        Returns:
            Preprocessed face image
        """
        model_name = os.path.basename(self.model_path)
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        
        param = {
            "org_img": frame,
            "bbox": face_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        
        if scale is None:
            param["crop"] = False
            
        img = self.image_cropper.crop(**param)
        return img
    
    def predict(self, face_img):
        """Predict if face is real or spoof
        
        Args:
            face_img: Preprocessed face image
            
        Returns:
            prediction: Anti-spoofing prediction results
        """
        # Prepare image
        img = self.test_transform(face_img)
        img = img.unsqueeze(0).to(self.device)
        
        # Load model from path
        model_name = os.path.basename(self.model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input)
        
        # Create and load model
        model = MiniFASNetV1SE(conv6_kernel=kernel_size).to(self.device)
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = next(keys)
        
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]  # Remove 'module.' prefix
                new_state_dict[name_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            result = model.forward(img)
            result = torch.nn.functional.softmax(result, dim=1).cpu().numpy()
        
        return result