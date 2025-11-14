"""Gemini (NanoBanana) Image Generation API Client for Empire.AI Music Video Maker

Provides access to Google's Gemini image generation capabilities
for creating and editing visual content.
"""

import os
import base64
from typing import Optional, List, Dict, Any, Literal, Union
import httpx
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    """Input image for generation or editing."""
    data: str = Field(description="Base64 encoded image data")
    mime_type: Literal["image/png", "image/jpeg"] = Field(description="Image MIME type")


class TextInput(BaseModel):
    """Text input for generation."""
    text: str = Field(description="Text prompt or instruction")


class GenerationConfig(BaseModel):
    """Configuration for image generation."""
    temperature: Optional[float] = Field(None, description="Temperature for generation (0.0-2.0)")
    top_p: Optional[float] = Field(None, description="Top-p for generation (0.0-1.0)")
    top_k: Optional[int] = Field(None, description="Top-k for generation")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    response_mime_type: Optional[str] = Field(None, description="Response MIME type")
    candidate_count: Optional[int] = Field(None, description="Number of candidates to generate")


class SafetySetting(BaseModel):
    """Safety setting for content filtering."""
    category: Literal[
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH", 
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT"
    ] = Field(description="Harm category")
    threshold: Literal[
        "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        "BLOCK_LOW_AND_ABOVE",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_NONE"
    ] = Field(description="Block threshold")


class GeneratedImage(BaseModel):
    """Generated image result."""
    data: str = Field(description="Base64 encoded image data")
    mime_type: str = Field(description="Image MIME type")
    width: Optional[int] = Field(None, description="Image width")
    height: Optional[int] = Field(None, description="Image height")


class TextToImageRequest(BaseModel):
    """Request for text-to-image generation."""
    prompt: str = Field(description="Text description of the image to generate")
    aspect_ratio: Optional[Literal[
        "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
    ]] = Field("1:1", description="Aspect ratio for the generated image")
    style: Optional[str] = Field(None, description="Style specification (e.g., 'photorealistic', 'artistic')")
    negative_prompt: Optional[str] = Field(None, description="What to avoid in the generation")
    generation_config: Optional[GenerationConfig] = Field(None, description="Generation configuration")
    safety_settings: Optional[List[SafetySetting]] = Field(None, description="Safety settings")


class ImageEditRequest(BaseModel):
    """Request for image editing."""
    image: ImageInput = Field(description="Input image to edit")
    prompt: str = Field(description="Editing instruction")
    mask: Optional[ImageInput] = Field(None, description="Optional mask for selective editing")
    generation_config: Optional[GenerationConfig] = Field(None, description="Generation configuration")
    safety_settings: Optional[List[SafetySetting]] = Field(None, description="Safety settings")


class MultiImageRequest(BaseModel):
    """Request for multi-image to image generation."""
    images: List[ImageInput] = Field(description="Input images")
    prompt: str = Field(description="Generation instruction")
    generation_config: Optional[GenerationConfig] = Field(None, description="Generation configuration")
    safety_settings: Optional[List[SafetySetting]] = Field(None, description="Safety settings")


class ImageGenerationResponse(BaseModel):
    """Response from image generation."""
    images: List[GeneratedImage] = Field(description="Generated images")
    prompt_feedback: Optional[Dict[str, Any]] = Field(None, description="Feedback on the prompt")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Usage statistics")


class GeminiImageClient:
    """Gemini (NanoBanana) image generation client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini image client.
        
        Args:
            api_key: Google API key. If not provided, will look for GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. "
                "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "gemini-2.5-flash-image"
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for image generation
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _prepare_image_input(self, image_path: str) -> ImageInput:
        """Prepare an image file for API input.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ImageInput with base64 encoded data
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Determine MIME type from file extension
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_path.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"
        else:
            raise ValueError("Unsupported image format. Use PNG or JPEG.")
        
        encoded_data = base64.b64encode(image_data).decode("utf-8")
        
        return ImageInput(data=encoded_data, mime_type=mime_type)
    
    def _prepare_image_from_bytes(self, image_bytes: bytes, mime_type: str) -> ImageInput:
        """Prepare image bytes for API input.
        
        Args:
            image_bytes: Raw image bytes
            mime_type: MIME type of the image
            
        Returns:
            ImageInput with base64 encoded data
        """
        encoded_data = base64.b64encode(image_bytes).decode("utf-8")
        return ImageInput(data=encoded_data, mime_type=mime_type)
    
    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the Gemini API."""
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}{endpoint}"
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for Gemini API errors
        if "error" in data:
            error_info = data["error"]
            raise Exception(f"Gemini API error: {error_info.get('message', 'Unknown error')}")
        
        return data
    
    def _build_content_parts(self, text: Optional[str] = None, 
                           images: Optional[List[ImageInput]] = None) -> List[Dict[str, Any]]:
        """Build content parts for the API request."""
        parts = []
        
        if text:
            parts.append({"text": text})
        
        if images:
            for image in images:
                parts.append({
                    "inline_data": {
                        "mime_type": image.mime_type,
                        "data": image.data
                    }
                })
        
        return parts
    
    def _parse_response(self, data: Dict[str, Any]) -> ImageGenerationResponse:
        """Parse the API response into structured format."""
        candidates = data.get("candidates", [])
        images = []
        
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            for part in parts:
                if "inline_data" in part:
                    inline_data = part["inline_data"]
                    image = GeneratedImage(
                        data=inline_data["data"],
                        mime_type=inline_data["mime_type"]
                    )
                    images.append(image)
        
        return ImageGenerationResponse(
            images=images,
            prompt_feedback=data.get("promptFeedback"),
            usage_metadata=data.get("usageMetadata")
        )
    
    async def generate_text_to_image(self, request: TextToImageRequest) -> ImageGenerationResponse:
        """Generate an image from text description.
        
        Args:
            request: Text-to-image generation parameters
            
        Returns:
            Generated image response
        """
        # Build prompt with aspect ratio and style
        full_prompt = request.prompt
        if request.aspect_ratio:
            full_prompt += f" [Aspect ratio: {request.aspect_ratio}]"
        if request.style:
            full_prompt += f" [Style: {request.style}]"
        if request.negative_prompt:
            full_prompt += f" [Avoid: {request.negative_prompt}]"
        
        parts = self._build_content_parts(text=full_prompt)
        
        payload = {
            "contents": [{
                "parts": parts
            }]
        }
        
        if request.generation_config:
            payload["generationConfig"] = request.generation_config.model_dump(exclude_none=True)
        
        if request.safety_settings:
            payload["safetySettings"] = [
                setting.model_dump() for setting in request.safety_settings
            ]
        
        endpoint = f"/models/{self.model}:generateContent"
        data = await self._make_request(endpoint, payload)
        
        return self._parse_response(data)
    
    async def edit_image(self, request: ImageEditRequest) -> ImageGenerationResponse:
        """Edit an existing image based on text instructions.
        
        Args:
            request: Image editing parameters
            
        Returns:
            Edited image response
        """
        images = [request.image]
        if request.mask:
            images.append(request.mask)
        
        parts = self._build_content_parts(text=request.prompt, images=images)
        
        payload = {
            "contents": [{
                "parts": parts
            }]
        }
        
        if request.generation_config:
            payload["generationConfig"] = request.generation_config.model_dump(exclude_none=True)
        
        if request.safety_settings:
            payload["safetySettings"] = [
                setting.model_dump() for setting in request.safety_settings
            ]
        
        endpoint = f"/models/{self.model}:generateContent"
        data = await self._make_request(endpoint, payload)
        
        return self._parse_response(data)
    
    async def generate_from_multiple_images(self, request: MultiImageRequest) -> ImageGenerationResponse:
        """Generate an image from multiple input images.
        
        Args:
            request: Multi-image generation parameters
            
        Returns:
            Generated image response
        """
        parts = self._build_content_parts(text=request.prompt, images=request.images)
        
        payload = {
            "contents": [{
                "parts": parts
            }]
        }
        
        if request.generation_config:
            payload["generationConfig"] = request.generation_config.model_dump(exclude_none=True)
        
        if request.safety_settings:
            payload["safetySettings"] = [
                setting.model_dump() for setting in request.safety_settings
            ]
        
        endpoint = f"/models/{self.model}:generateContent"
        data = await self._make_request(endpoint, payload)
        
        return self._parse_response(data)
    
    async def create_artist_concept_image(self, artist_name: str, song_title: str, 
                                        concept: str, style: str = "cinematic",
                                        aspect_ratio: str = "16:9") -> GeneratedImage:
        """Create a concept image for an artist and song.
        
        Args:
            artist_name: Name of the artist
            song_title: Title of the song
            concept: Visual concept description
            style: Visual style (default: cinematic)
            aspect_ratio: Image aspect ratio
            
        Returns:
            Generated concept image
        """
        prompt = (
            f"Create a {style} concept image for '{song_title}' by {artist_name}. "
            f"Visual concept: {concept}. "
            f"High quality, professional music video aesthetic, "
            f"dramatic lighting, rich colors."
        )
        
        request = TextToImageRequest(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            style=style
        )
        
        response = await self.generate_text_to_image(request)
        
        if not response.images:
            raise Exception("No images generated")
        
        return response.images[0]
    
    async def enhance_artist_photo(self, image_path: str, enhancement: str) -> GeneratedImage:
        """Enhance an artist photo with specific modifications.
        
        Args:
            image_path: Path to the artist photo
            enhancement: Description of enhancements to apply
            
        Returns:
            Enhanced image
        """
        image_input = self._prepare_image_input(image_path)
        
        prompt = f"Enhance this artist photo: {enhancement}. Maintain the person's likeness while applying the requested changes."
        
        request = ImageEditRequest(
            image=image_input,
            prompt=prompt
        )
        
        response = await self.edit_image(request)
        
        if not response.images:
            raise Exception("No images generated")
        
        return response.images[0]
    
    def save_image(self, image: GeneratedImage, output_path: str):
        """Save a generated image to file.
        
        Args:
            image: Generated image to save
            output_path: Path where to save the image
        """
        image_bytes = base64.b64decode(image.data)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
