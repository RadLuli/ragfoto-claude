import cv2
import numpy as np
from PIL import Image, ImageStat
from typing import Dict, Any, Tuple
import os

class ImageAnalyzer:
    """Analyzes photography based on various quality metrics."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze an image file and return quality metrics."""
        # Load image in different formats for different analyses
        cv_image = cv2.imread(image_path)
        pil_image = Image.open(image_path)
        
        # Get basic image information
        width, height = pil_image.size
        aspect_ratio = width / height
        
        # Calculate brightness
        stat = ImageStat.Stat(pil_image)
        brightness = sum(stat.mean) / len(stat.mean)
        
        # Calculate contrast
        contrast = self._calculate_contrast(cv_image)
        
        # Detect rule of thirds
        rule_of_thirds_score = self._analyze_rule_of_thirds(cv_image)
        
        # Calculate sharpness
        sharpness = self._calculate_sharpness(cv_image)
        
        # Analyze color balance
        color_balance = self._analyze_color_balance(pil_image)
        
        # Detect faces (for portrait assessment)
        face_count = self._detect_faces(cv_image)
        
        return {
            "dimensions": {"width": width, "height": height},
            "aspect_ratio": aspect_ratio,
            "brightness": brightness,  # 0-255 scale
            "contrast": contrast,  # 0-1 scale
            "rule_of_thirds": rule_of_thirds_score,  # 0-1 scale
            "sharpness": sharpness,  # Higher is sharper
            "color_balance": color_balance,  # Dictionary of color metrics
            "faces": face_count  # Number of faces detected
        }
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate the contrast of an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.std() / 255
    
    def _analyze_rule_of_thirds(self, image: np.ndarray) -> float:
        """Analyze adherence to rule of thirds."""
        # Use edge detection to find significant elements
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Define thirds grid
        h, w = edges.shape
        h_third, w_third = h // 3, w // 3
        
        # Check edge density around rule of thirds intersections
        intersection_regions = [
            (w_third, h_third),
            (w_third * 2, h_third),
            (w_third, h_third * 2),
            (w_third * 2, h_third * 2)
        ]
        
        # Calculate edge density at intersections vs. overall
        total_edges = np.sum(edges > 0)
        if total_edges == 0:
            return 0.0
        
        intersection_edge_sum = 0
        region_size = min(h, w) // 10
        
        for x, y in intersection_regions:
            x1, x2 = max(0, x - region_size), min(w, x + region_size)
            y1, y2 = max(0, y - region_size), min(h, y + region_size)
            region = edges[y1:y2, x1:x2]
            intersection_edge_sum += np.sum(region > 0)
        
        # Calculate score (how much of the edge content is near rule of thirds points)
        region_pixels = 4 * (region_size * 2) ** 2
        total_pixels = h * w
        
        expected_random_proportion = region_pixels / total_pixels
        actual_proportion = intersection_edge_sum / total_edges
        
        # Normalize score to 0-1
        score = min(1.0, actual_proportion / (expected_random_proportion * 2))
        return score
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return lap.var()
    
    def _analyze_color_balance(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze color balance and distribution."""
        # Convert to RGB if not already
        rgb_image = image.convert('RGB')
        
        # Get histograms for each channel
        r_hist = rgb_image.histogram()[0:256]
        g_hist = rgb_image.histogram()[256:512]
        b_hist = rgb_image.histogram()[512:768]
        
        # Calculate average values for each channel
        r_avg = sum(i * count for i, count in enumerate(r_hist)) / sum(r_hist)
        g_avg = sum(i * count for i, count in enumerate(g_hist)) / sum(g_hist)
        b_avg = sum(i * count for i, count in enumerate(b_hist)) / sum(b_hist)
        
        # Calculate color balance (ideally they should be roughly equal)
        avg = (r_avg + g_avg + b_avg) / 3
        r_balance = r_avg / avg if avg > 0 else 1
        g_balance = g_avg / avg if avg > 0 else 1
        b_balance = b_avg / avg if avg > 0 else 1
        
        # Calculate color variance (measure of color richness)
        r_var = sum((i - r_avg) ** 2 * count for i, count in enumerate(r_hist)) / sum(r_hist)
        g_var = sum((i - g_avg) ** 2 * count for i, count in enumerate(g_hist)) / sum(g_hist)
        b_var = sum((i - b_avg) ** 2 * count for i, count in enumerate(b_hist)) / sum(b_hist)
        
        return {
            "channel_avg": {"red": r_avg, "green": g_avg, "blue": b_avg},
            "balance": {"red": r_balance, "green": g_balance, "blue": b_balance},
            "variance": {"red": r_var, "green": g_var, "blue": b_var}
        }
    
    def _detect_faces(self, image: np.ndarray) -> int:
        """Detect the number of faces in an image."""
        # Use a pre-trained face detector from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces)
    
    def save_analysis_visualization(self, image_path: str, output_path: str) -> str:
        """Create a visualization of the analysis and save it."""
        # Load image
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Draw rule of thirds grid
        viz_image = image.copy()
        h_third, w_third = h // 3, w // 3
        
        # Draw horizontal lines
        cv2.line(viz_image, (0, h_third), (w, h_third), (0, 255, 255), 1)
        cv2.line(viz_image, (0, h_third * 2), (w, h_third * 2), (0, 255, 255), 1)
        
        # Draw vertical lines
        cv2.line(viz_image, (w_third, 0), (w_third, h), (0, 255, 255), 1)
        cv2.line(viz_image, (w_third * 2, 0), (w_third * 2, h), (0, 255, 255), 1)
        
        # Draw rule of thirds intersection points
        intersection_points = [
            (w_third, h_third),
            (w_third * 2, h_third),
            (w_third, h_third * 2),
            (w_third * 2, h_third * 2)
        ]
        for point in intersection_points:
            cv2.circle(viz_image, point, 5, (0, 0, 255), -1)
        
        # Detect faces and draw rectangles around them
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w_face, h_face) in faces:
            cv2.rectangle(viz_image, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
        
        # Add text with some key metrics
        analysis = self.analyze_image(image_path)
        bright_text = f"Brightness: {analysis['brightness']:.1f}"
        contrast_text = f"Contrast: {analysis['contrast']:.2f}"
        sharp_text = f"Sharpness: {analysis['sharpness']:.0f}"
        
        cv2.putText(viz_image, bright_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, contrast_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz_image, sharp_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, viz_image)
        
        return output_path
