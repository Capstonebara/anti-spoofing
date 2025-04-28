import cv2
import torch
from facenet_pytorch import MTCNN

class FaceDetector:
    def __init__(self):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Face Detector using device: {self.device}")
        
        # Initialize MTCNN
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=[0.6, 0.7, 0.7],
            min_face_size=20
        )
    
    def detect_faces(self, frame):
        """Detect faces in the given frame
        
        Args:
            frame: BGR frame from opencv
            
        Returns:
            boxes: face bounding boxes
            probs: confidence scores
        """
        # Convert frame from BGR to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = self.detector.detect(rgb_frame)
        
        return boxes, probs