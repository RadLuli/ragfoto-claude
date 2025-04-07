import requests
import json
from typing import Dict, Any, List
import os
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

class LLMService:
    """Service for interacting with the LLM for photo assessment."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize LLM service with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize LLM
        self.llm = Ollama(
            model=self.config["model"]["local_model"],
            temperature=self.config["model"]["temperature"]
        )
    
    def generate_assessment(self, image_analysis: Dict[str, Any], query_text: str, 
                         reference_content: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a photo assessment using the LLM."""
        # Create context by combining analysis and reference content
        context = self._create_context(image_analysis, reference_content)
        
        # Prepare prompt for assessment
        prompt = self._create_assessment_prompt(context, query_text, image_analysis)
        
        # Query the LLM
        response = self._query_llm(prompt)
        
        # Parse the response
        parsed_response = self._parse_assessment_response(response)
        
        return parsed_response
    
    def _create_context(self, image_analysis: Dict[str, Any], reference_content: List[Dict[str, str]]) -> str:
        """Create context by combining image analysis and reference content."""
        # Format image analysis as readable text
        analysis_text = f"""
        Image Analysis:
        - Dimensions: {image_analysis['dimensions']['width']}x{image_analysis['dimensions']['height']}
        - Aspect Ratio: {image_analysis['aspect_ratio']:.2f}
        - Brightness: {image_analysis['brightness']:.2f} (0-255 scale, ideal range 80-180)
        - Contrast: {image_analysis['contrast']:.2f} (0-1 scale, ideal range 0.4-0.7)
        - Rule of Thirds Adherence: {image_analysis['rule_of_thirds']:.2f} (0-1 scale, higher is better)
        - Sharpness: {image_analysis['sharpness']:.2f} (higher is sharper)
        - Color Balance (RGB): R={image_analysis['color_balance']['balance']['red']:.2f}, G={image_analysis['color_balance']['balance']['green']:.2f}, B={image_analysis['color_balance']['balance']['blue']:.2f} (ideal is close to 1.0 for each)
        - Faces Detected: {image_analysis['faces']}
        """
        
        # Add reference content
        reference_text = "\n\nReference Content:\n"
        for i, ref in enumerate(reference_content):
            reference_text += f"\n--- Reference {i+1} (Source: {ref['source']}) ---\n{ref['content']}\n"
        
        return analysis_text + reference_text
    
    def _create_assessment_prompt(self, context: str, query_text: str, image_analysis: Dict[str, Any]) -> str:
        """Create a prompt for the LLM to assess the photo."""
        scoring_criteria = self.config["scoring"]["criteria"]
        
        prompt = f"""You are a professional photography teacher providing feedback to students on their photos.

CONTEXT INFORMATION:
{context}

TASK:
Analyze the photo based on the technical analysis provided and the reference content. 

1. Provide an overall assessment of the photo's quality and composition.
2. Score the photo on a scale of 1-5 stars based on these criteria: {', '.join(scoring_criteria)}.
3. Provide specific suggestions for improvement.
4. Recommend technical adjustments that could enhance the photo.

Format your response in JSON with these fields:
- overall_assessment: A paragraph assessing the photo
- score: A number from 1-5 (can include half points like 3.5)
- criteria_scores: Individual scores for each criterion
- suggestions: An array of specific improvement suggestions
- technical_adjustments: An array of specific technical adjustments

Student query or additional context: {query_text}

JSON RESPONSE:
"""
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt."""
        try:
            response = self.llm(prompt)
            return response
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return """{"overall_assessment": "Unable to analyze the photo due to a technical issue. Please try again.",
                    "score": 0,
                    "criteria_scores": {},
                    "suggestions": ["Try uploading the photo again."],
                    "technical_adjustments": []}"""
    
    def _parse_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract the structured assessment."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            # Fallback: try to parse the whole response as JSON
            return json.loads(response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            # Return a default structure
            return {
                "overall_assessment": "The system couldn't parse the analysis properly. Here's the raw assessment: " + response[:500],
                "score": 3,
                "criteria_scores": {},
                "suggestions": ["Please try again."],
                "technical_adjustments": []
            }
    
    def generate_suggestions(self, assessment: Dict[str, Any], image_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific enhancement suggestions based on the assessment."""
        # Create a prompt for enhancement suggestions
        prompt = f"""
        Based on this photo assessment and technical analysis, provide SPECIFIC technical adjustments to enhance the image:
        
        Assessment: {assessment['overall_assessment']}
        Score: {assessment['score']} out of 5
        Technical Analysis:
        - Brightness: {image_analysis['brightness']:.2f} (ideal range 80-180)
        - Contrast: {image_analysis['contrast']:.2f} (ideal range 0.4-0.7)
        - Rule of Thirds: {image_analysis['rule_of_thirds']:.2f}
        - Sharpness: {image_analysis['sharpness']:.2f}
        
        Give 3-5 SPECIFIC technical adjustments that can be directly applied to the image.
        Format your response as a JSON array of strings, each suggestion being a clear instruction for image enhancement.
        """
        
        # Query the LLM
        response = self._query_llm(prompt)
        
        try:
            # Extract JSON array from response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            # Fallback: return suggestions from assessment
            return assessment.get("suggestions", ["Adjust brightness if needed.", "Consider rule of thirds.", "Check focus."])
        except Exception as e:
            print(f"Error parsing suggestions: {e}")
            return ["Adjust brightness if needed.", "Consider rule of thirds.", "Check focus."]
