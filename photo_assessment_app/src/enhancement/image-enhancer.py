import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from typing import Dict, Any, List, Tuple

class ImageEnhancer:
    """Enhances photos based on analysis and suggestions."""
    
    def __init__(self):
        """Initialize the enhancer."""
        pass
    
    def enhance_image(self, image_path: str, analysis: Dict[str, Any], suggestions: List[str], output_path: str) -> str:
        """Enhance an image based on analysis and suggestions."""
        # Load image
        pil_image = Image.open(image_path)
        
        # Parse suggestions to determine what adjustments to make
        adjustments = self._parse_suggestions(suggestions, analysis)
        
        # Apply enhancements
        enhanced_image = self._apply_enhancements(pil_image, adjustments, analysis)
        
        # Save enhanced image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        enhanced_image.save(output_path, quality=95)
        
        return output_path
    
    def _parse_suggestions(self, suggestions: List[str], analysis: Dict[str, Any]) -> Dict[str, float]:
        """Parse suggestions to determine adjustment values."""
        adjustments = {
            "brightness": 0,
            "contrast": 0,
            "color": 0,
            "sharpness": 0,
            "warmth": 0,  # Color temperature
            "crop_rule_thirds": False
        }
        
        # Keywords to look for in suggestions
        brightness_keywords = ["bright", "dark", "exposure", "illumina", "luz", "escur", "exposição"]
        contrast_keywords = ["contrast", "flat", "contraste", "plano"]
        color_keywords = ["saturation", "vibrant", "vivid", "colorful", "saturação", "vibrante", "colorido"]
        sharpness_keywords = ["sharp", "blur", "focus", "nitid", "foco", "desfoque"]
        warmth_keywords = ["warm", "cool", "temperature", "quente", "frio", "temperatura"]
        composition_keywords = ["composition", "rule of thirds", "crop", "composição", "regra dos terços", "cortar"]
        
        # Current values from analysis
        current_brightness = analysis.get("brightness", 128) / 255  # Normalize to 0-1
        current_contrast = analysis.get("contrast", 0.5)
        current_rule_thirds = analysis.get("rule_of_thirds", 0.5)
        
        # Process each suggestion
        for suggestion in suggestions:
            suggestion = suggestion.lower()
            
            # Check brightness adjustments
            if any(keyword in suggestion for keyword in brightness_keywords):
                if any(word in suggestion for word in ["increase", "more", "brighter", "higher", "aumentar", "mais", "maior"]):
                    adjustments["brightness"] = max(0.1, min(0.5, 1 - current_brightness))
                elif any(word in suggestion for word in ["decrease", "less", "darker", "lower", "diminuir", "menos", "menor"]):
                    adjustments["brightness"] = max(-0.5, min(-0.1, 0 - current_brightness))
            
            # Check contrast adjustments
            if any(keyword in suggestion for keyword in contrast_keywords):
                if any(word in suggestion for word in ["increase", "more", "higher", "aumentar", "mais", "maior"]):
                    adjustments["contrast"] = max(0.1, min(0.7, 1 - current_contrast))
                elif any(word in suggestion for word in ["decrease", "less", "lower", "diminuir", "menos", "menor"]):
                    adjustments["contrast"] = max(-0.3, min(-0.1, 0 - current_contrast))
            
            # Check color adjustments
            if any(keyword in suggestion for keyword in color_keywords):
                if any(word in suggestion for word in ["increase", "more", "vibrant", "colorful", "aumentar", "mais", "vibrante"]):
                    adjustments["color"] = 0.3
                elif any(word in suggestion for word in ["decrease", "less", "muted", "diminuir", "menos"]):
                    adjustments["color"] = -0.2
            
            # Check sharpness adjustments
            if any(keyword in suggestion for keyword in sharpness_keywords):
                if any(word in suggestion for word in ["increase", "more", "sharper", "aumentar", "mais", "maior"]):
                    adjustments["sharpness"] = 0.5
                elif any(word in suggestion for word in ["decrease", "less", "softer", "diminuir", "menos", "menor"]):
                    adjustments["sharpness"] = -0.2
            
            # Check warmth adjustments
            if any(keyword in suggestion for keyword in warmth_keywords):
                if any(word in suggestion for word in ["warmer", "increase", "more", "mais quente", "aumentar"]):
                    adjustments["warmth"] = 0.2
                elif any(word in suggestion for word in ["cooler", "decrease", "less", "mais frio", "diminuir"]):
                    adjustments["warmth"] = -0.2
            
            # Check composition/crop adjustments
            if any(keyword in suggestion for keyword in composition_keywords):
                if "rule of thirds" in suggestion or "regra dos terços" in suggestion:
                    if current_rule_thirds < 0.4:  # Only crop if rule of thirds score is low
                        adjustments["crop_rule_thirds"] = True
        
        return adjustments
    
    def _apply_enhancements(self, image: Image.Image, adjustments: Dict[str, float], analysis: Dict[str, Any]) -> Image.Image:
        """Apply enhancements to the image."""
        # Apply crop for rule of thirds if needed
        if adjustments["crop_rule_thirds"]:
            image = self._crop_for_rule_of_thirds(image)
        
        # Apply brightness adjustment
        if adjustments["brightness"] != 0:
            enhancer = ImageEnhance.Brightness(image)
            factor = 1.0 + adjustments["brightness"]
            image = enhancer.enhance(factor)
        
        # Apply contrast adjustment
        if adjustments["contrast"] != 0:
            enhancer = ImageEnhance.Contrast(image)
            factor = 1.0 + adjustments["contrast"]
            image = enhancer.enhance(factor)
        
        # Apply color/saturation adjustment
        if adjustments["color"] != 0:
            enhancer = ImageEnhance.Color(image)
            factor = 1.0 + adjustments["color"]
            image = enhancer.enhance(factor)
        
        # Apply sharpness adjustment
        if adjustments["sharpness"] != 0:
            enhancer = ImageEnhance.Sharpness(image)
            factor = 1.0 + adjustments["sharpness"]
            image = enhancer.enhance(factor)
        
        # Apply warmth adjustment (color temperature)
        if adjustments["warmth"] != 0:
            image = self._adjust_warmth(image, adjustments["warmth"])
        
        return image
    
    def _adjust_warmth(self, image: Image.Image, adjustment: float) -> Image.Image:
        """Adjust the color temperature (warmth) of an image."""
        # Convert to numpy array for easier manipulation
        img_array = np.array(image)
        
        # RGB adjustments for warmth
        # Increase red and decrease blue for warmth, and vice versa for coolness
        if adjustment > 0:  # Warmer
            # Enhance red, reduce blue
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + adjustment), 0, 255)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - adjustment/2), 0, 255)
        else:  # Cooler
            # Enhance blue, reduce red
            adjustment = abs(adjustment)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 + adjustment), 0, 255)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 - adjustment/2), 0, 255)
        
        # Convert back to PIL Image
        return Image.fromarray(img_array)
    
mkdir -p data/{pdfs,ebooks,web_content}
mkdir -p src/{document_processing,image_analysis,translation,enhancement,web}
