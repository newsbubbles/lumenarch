"""Veo 3.1 Video Generation API Client for Empire.AI Music Video Maker

Provides access to Google's Veo video generation capabilities
for creating high-quality music video content.
"""

import os
import base64
import asyncio
from typing import Optional, List, Dict, Any, Literal, Union
import httpx
from pydantic import BaseModel, Field


class VideoInput(BaseModel):
    """Input video for generation or extension."""
    data: str = Field(description="Base64 encoded video data")
    mime_type: str = Field(description="Video MIME type")


class ImageInput(BaseModel):
    """Input image for video generation."""
    data: str = Field(description="Base64 encoded image data")
    mime_type: Literal["image/png", "image/jpeg"] = Field(description="Image MIME type")


class VideoGenerationConfig(BaseModel):
    """Configuration for video generation."""
    resolution: Optional[Literal["720p", "1080p"]] = Field("720p", description="Video resolution")
    aspect_ratio: Optional[Literal["16:9", "9:16"]] = Field("16:9", description="Video aspect ratio")
    duration: Optional[Literal[4, 6, 8]] = Field(8, description="Video duration in seconds")
    frame_rate: Optional[int] = Field(24, description="Frame rate (typically 24fps)")
    person_generation: Optional[bool] = Field(True, description="Allow person generation")


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


class GeneratedVideo(BaseModel):
    """Generated video result."""
    data: str = Field(description="Base64 encoded video data")
    mime_type: str = Field(description="Video MIME type")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    width: Optional[int] = Field(None, description="Video width")
    height: Optional[int] = Field(None, description="Video height")
    frame_rate: Optional[float] = Field(None, description="Video frame rate")


class OperationStatus(BaseModel):
    """Status of a long-running operation."""
    name: str = Field(description="Operation name")
    done: bool = Field(description="Whether operation is complete")
    response: Optional[Dict[str, Any]] = Field(None, description="Response data if complete")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information if failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Operation metadata")


class TextToVideoRequest(BaseModel):
    """Request for text-to-video generation."""
    prompt: str = Field(description="Text description of the video to generate")
    config: Optional[VideoGenerationConfig] = Field(None, description="Generation configuration")
    safety_settings: Optional[List[SafetySetting]] = Field(None, description="Safety settings")


class ImageToVideoRequest(BaseModel):
    """Request for image-to-video generation."""
    image: ImageInput = Field(description="Input image")
    prompt: str = Field(description="Text description for video generation")
    config: Optional[VideoGenerationConfig] = Field(None, description="Generation configuration")
    safety_settings: Optional[List[SafetySetting]] = Field(None, description="Safety settings")


class VideoExtensionRequest(BaseModel):
    """Request for extending an existing video."""
    video: VideoInput = Field(description="Input video to extend (must be Veo-generated)")
    prompt: str = Field(description="Description for the extension")
    config: Optional[VideoGenerationConfig] = Field(None, description="Generation configuration")
    safety_settings: Optional[List[SafetySetting]] = Field(None, description="Safety settings")


class FrameDirectedRequest(BaseModel):
    """Request for frame-directed video generation."""
    first_frame: ImageInput = Field(description="First frame image")
    last_frame: ImageInput = Field(description="Last frame image")
    prompt: str = Field(description="Text description for video generation")
    config: Optional[VideoGenerationConfig] = Field(None, description="Generation configuration")
    safety_settings: Optional[List[SafetySetting]] = Field(None, description="Safety settings")


class VideoGenerationResponse(BaseModel):
    """Response from video generation."""
    operation_name: str = Field(description="Operation name for status tracking")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")


class VideoGenerationResult(BaseModel):
    """Final result of video generation."""
    video: GeneratedVideo = Field(description="Generated video")
    prompt_feedback: Optional[Dict[str, Any]] = Field(None, description="Feedback on the prompt")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Usage statistics")


class VeoVideoClient:
    """Veo 3.1 video generation client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Veo video client.
        
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
        self.model = "veo-3.1-generate-preview"
        self.client = httpx.AsyncClient(timeout=600.0)  # 10 minute timeout for video generation
    
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
    
    def _prepare_video_input(self, video_path: str) -> VideoInput:
        """Prepare a video file for API input.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoInput with base64 encoded data
        """
        with open(video_path, "rb") as f:
            video_data = f.read()
        
        # Determine MIME type (assuming MP4 for Veo videos)
        mime_type = "video/mp4"
        
        encoded_data = base64.b64encode(video_data).decode("utf-8")
        
        return VideoInput(data=encoded_data, mime_type=mime_type)
    
    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the Veo API."""
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}{endpoint}"
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if "error" in data:
            error_info = data["error"]
            raise Exception(f"Veo API error: {error_info.get('message', 'Unknown error')}")
        
        return data
    
    async def _get_operation_status(self, operation_name: str) -> OperationStatus:
        """Get the status of a long-running operation."""
        headers = {"x-goog-api-key": self.api_key}
        
        url = f"{self.base_url}/{operation_name}"
        response = await self.client.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        return OperationStatus(
            name=data["name"],
            done=data.get("done", False),
            response=data.get("response"),
            error=data.get("error"),
            metadata=data.get("metadata")
        )
    
    def _build_content_parts(self, text: Optional[str] = None,
                           images: Optional[List[ImageInput]] = None,
                           videos: Optional[List[VideoInput]] = None) -> List[Dict[str, Any]]:
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
        
        if videos:
            for video in videos:
                parts.append({
                    "inline_data": {
                        "mime_type": video.mime_type,
                        "data": video.data
                    }
                })
        
        return parts
    
    def _parse_result(self, response_data: Dict[str, Any]) -> VideoGenerationResult:
        """Parse the final result from operation response."""
        candidates = response_data.get("candidates", [])
        
        if not candidates:
            raise Exception("No video generated")
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        for part in parts:
            if "inline_data" in part:
                inline_data = part["inline_data"]
                video = GeneratedVideo(
                    data=inline_data["data"],
                    mime_type=inline_data["mime_type"]
                )
                
                return VideoGenerationResult(
                    video=video,
                    prompt_feedback=response_data.get("promptFeedback"),
                    usage_metadata=response_data.get("usageMetadata")
                )
        
        raise Exception("No video data found in response")
    
    async def generate_text_to_video(self, request: TextToVideoRequest) -> VideoGenerationResponse:
        """Generate a video from text description.
        
        Args:
            request: Text-to-video generation parameters
            
        Returns:
            Video generation response with operation name
        """
        parts = self._build_content_parts(text=request.prompt)
        
        payload = {
            "contents": [{
                "parts": parts
            }]
        }
        
        if request.config:
            generation_config = {}
            if request.config.resolution:
                generation_config["resolution"] = request.config.resolution
            if request.config.aspect_ratio:
                generation_config["aspectRatio"] = request.config.aspect_ratio
            if request.config.duration:
                generation_config["duration"] = f"{request.config.duration}s"
            if request.config.person_generation is not None:
                generation_config["personGeneration"] = request.config.person_generation
            
            if generation_config:
                payload["generationConfig"] = generation_config
        
        if request.safety_settings:
            payload["safetySettings"] = [
                setting.model_dump() for setting in request.safety_settings
            ]
        
        endpoint = f"/models/{self.model}:predictLongRunning"
        data = await self._make_request(endpoint, payload)
        
        return VideoGenerationResponse(
            operation_name=data["name"],
            estimated_completion_time=data.get("metadata", {}).get("estimatedCompletionTime")
        )
    
    async def generate_image_to_video(self, request: ImageToVideoRequest) -> VideoGenerationResponse:
        """Generate a video from an input image.
        
        Args:
            request: Image-to-video generation parameters
            
        Returns:
            Video generation response with operation name
        """
        parts = self._build_content_parts(text=request.prompt, images=[request.image])
        
        payload = {
            "contents": [{
                "parts": parts
            }]
        }
        
        if request.config:
            generation_config = {}
            if request.config.resolution:
                generation_config["resolution"] = request.config.resolution
            if request.config.aspect_ratio:
                generation_config["aspectRatio"] = request.config.aspect_ratio
            if request.config.duration:
                generation_config["duration"] = f"{request.config.duration}s"
            if request.config.person_generation is not None:
                generation_config["personGeneration"] = request.config.person_generation
            
            if generation_config:
                payload["generationConfig"] = generation_config
        
        if request.safety_settings:
            payload["safetySettings"] = [
                setting.model_dump() for setting in request.safety_settings
            ]
        
        endpoint = f"/models/{self.model}:predictLongRunning"
        data = await self._make_request(endpoint, payload)
        
        return VideoGenerationResponse(
            operation_name=data["name"],
            estimated_completion_time=data.get("metadata", {}).get("estimatedCompletionTime")
        )
    
    async def extend_video(self, request: VideoExtensionRequest) -> VideoGenerationResponse:
        """Extend an existing Veo-generated video.
        
        Args:
            request: Video extension parameters
            
        Returns:
            Video generation response with operation name
        """
        parts = self._build_content_parts(text=request.prompt, videos=[request.video])
        
        payload = {
            "contents": [{
                "parts": parts
            }]
        }
        
        if request.config:
            generation_config = {}
            if request.config.resolution:
                generation_config["resolution"] = request.config.resolution
            if request.config.aspect_ratio:
                generation_config["aspectRatio"] = request.config.aspect_ratio
            # Duration must be 8s for extensions
            generation_config["duration"] = "8s"
            if request.config.person_generation is not None:
                generation_config["personGeneration"] = request.config.person_generation
            
            payload["generationConfig"] = generation_config
        
        if request.safety_settings:
            payload["safetySettings"] = [
                setting.model_dump() for setting in request.safety_settings
            ]
        
        endpoint = f"/models/{self.model}:predictLongRunning"
        data = await self._make_request(endpoint, payload)
        
        return VideoGenerationResponse(
            operation_name=data["name"],
            estimated_completion_time=data.get("metadata", {}).get("estimatedCompletionTime")
        )
    
    async def generate_frame_directed(self, request: FrameDirectedRequest) -> VideoGenerationResponse:
        """Generate a video with specified first and last frames.
        
        Args:
            request: Frame-directed generation parameters
            
        Returns:
            Video generation response with operation name
        """
        parts = self._build_content_parts(
            text=request.prompt, 
            images=[request.first_frame, request.last_frame]
        )
        
        payload = {
            "contents": [{
                "parts": parts
            }]
        }
        
        if request.config:
            generation_config = {}
            if request.config.resolution:
                generation_config["resolution"] = request.config.resolution
            if request.config.aspect_ratio:
                generation_config["aspectRatio"] = request.config.aspect_ratio
            # Duration must be 8s for frame-directed generation
            generation_config["duration"] = "8s"
            if request.config.person_generation is not None:
                generation_config["personGeneration"] = request.config.person_generation
            
            payload["generationConfig"] = generation_config
        
        if request.safety_settings:
            payload["safetySettings"] = [
                setting.model_dump() for setting in request.safety_settings
            ]
        
        endpoint = f"/models/{self.model}:predictLongRunning"
        data = await self._make_request(endpoint, payload)
        
        return VideoGenerationResponse(
            operation_name=data["name"],
            estimated_completion_time=data.get("metadata", {}).get("estimatedCompletionTime")
        )
    
    async def wait_for_completion(self, operation_name: str, 
                                poll_interval: int = 30, 
                                max_wait_time: int = 600) -> VideoGenerationResult:
        """Wait for a video generation operation to complete.
        
        Args:
            operation_name: Operation name from generation response
            poll_interval: Seconds between status checks
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            Final video generation result
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            status = await self._get_operation_status(operation_name)
            
            if status.done:
                if status.error:
                    raise Exception(f"Video generation failed: {status.error}")
                
                if status.response:
                    return self._parse_result(status.response)
                else:
                    raise Exception("Operation completed but no response data")
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait_time:
                raise Exception(f"Video generation timed out after {max_wait_time} seconds")
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def create_music_video_clip(self, artist_name: str, song_title: str,
                                    concept: str, style: str = "cinematic",
                                    duration: int = 8, aspect_ratio: str = "16:9") -> VideoGenerationResult:
        """Create a music video clip for an artist and song.
        
        Args:
            artist_name: Name of the artist
            song_title: Title of the song
            concept: Visual concept description
            style: Visual style (default: cinematic)
            duration: Video duration in seconds
            aspect_ratio: Video aspect ratio
            
        Returns:
            Generated music video clip
        """
        prompt = (
            f"Create a {style} music video clip for '{song_title}' by {artist_name}. "
            f"Visual concept: {concept}. "
            f"High production value, professional music video aesthetic, "
            f"dynamic camera movements, rich colors, dramatic lighting. "
            f"Sync visual rhythm with musical energy."
        )
        
        config = VideoGenerationConfig(
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution="1080p"
        )
        
        request = TextToVideoRequest(
            prompt=prompt,
            config=config
        )
        
        response = await self.generate_text_to_video(request)
        return await self.wait_for_completion(response.operation_name)
    
    def save_video(self, video: GeneratedVideo, output_path: str):
        """Save a generated video to file.
        
        Args:
            video: Generated video to save
            output_path: Path where to save the video
        """
        video_bytes = base64.b64decode(video.data)
        with open(output_path, "wb") as f:
            f.write(video_bytes)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
