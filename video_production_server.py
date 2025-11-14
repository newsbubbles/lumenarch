"""Video Production MCP Server for Empire.AI

Provides video generation capabilities through Veo 3.1 API.
Exposes tools for creating high-quality music videos, extending sequences, and managing video production workflows.
"""

import logging
import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Any, Optional, List, Dict, Literal
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context

# Import our API client
from clients.veo.client import (
    VeoVideoClient, TextToVideoRequest, ImageToVideoRequest,
    VideoExtensionRequest, FrameDirectedRequest, VideoGenerationResponse,
    VideoGenerationResult, GeneratedVideo, VideoInput, ImageInput,
    VideoGenerationConfig, SafetySetting, OperationStatus
)

def setup_mcp_logging():
    """Standard logging setup for MCP servers in testing environment."""
    logger_name = os.getenv("LOGGER_NAME", __name__)
    logger_path = os.getenv("LOGGER_PATH")
    
    logger = logging.getLogger(logger_name)
    
    # Only add handler if we have a log path (testing environment)
    if logger_path and not logger.handlers:
        handler = logging.FileHandler(logger_path, mode='a')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# Set up logging
logger = setup_mcp_logging()
logger.info("Video Production MCP server starting")

# Request/Response Models for MCP Tools

class MusicVideoClipRequest(BaseModel):
    """Request for generating music video clips."""
    artist_name: str = Field(description="Artist name")
    song_title: str = Field(description="Song title")
    concept_description: str = Field(description="Visual concept for the video")
    style: Literal["cinematic", "artistic", "performance", "narrative", "abstract", "retro"] = Field(
        "cinematic", description="Visual style for the video"
    )
    mood: Optional[str] = Field(None, description="Emotional mood (e.g., energetic, melancholic, dramatic)")
    duration: Literal[4, 6, 8] = Field(8, description="Video duration in seconds")
    aspect_ratio: Literal["16:9", "9:16"] = Field("16:9", description="Video aspect ratio")
    resolution: Literal["720p", "1080p"] = Field("1080p", description="Video resolution")
    include_performance: bool = Field(True, description="Include performance elements")
    color_palette: Optional[str] = Field(None, description="Preferred color palette")
    camera_style: Optional[str] = Field(None, description="Camera movement style (static, dynamic, handheld)")
    sync_to_music: bool = Field(True, description="Synchronize visual rhythm with musical energy")

class MusicVideoClipResponse(BaseModel):
    """Response for music video clip generation."""
    artist_name: str = Field(description="Artist name")
    song_title: str = Field(description="Song title")
    operation_name: str = Field(description="Video generation operation ID")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    concept_applied: str = Field(description="Concept that was applied")
    style_applied: str = Field(description="Style that was applied")
    generation_prompt: str = Field(description="Final prompt used for generation")
    video_config: Dict[str, Any] = Field(description="Video configuration used")

class VideoGenerationStatusRequest(BaseModel):
    """Request for checking video generation status."""
    operation_name: str = Field(description="Video generation operation ID")

class VideoGenerationStatusResponse(BaseModel):
    """Response for video generation status check."""
    operation_name: str = Field(description="Video generation operation ID")
    status: Literal["pending", "running", "completed", "failed"] = Field(description="Generation status")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[VideoGenerationResult] = Field(None, description="Video result if completed")

class VideoSequenceExtensionRequest(BaseModel):
    """Request for extending existing video sequences."""
    video_path: str = Field(description="Path to the video file to extend")
    extension_concept: str = Field(description="Concept for the video extension")
    maintain_style: bool = Field(True, description="Maintain the style of the original video")
    transition_type: Literal["smooth", "cut", "fade", "creative"] = Field(
        "smooth", description="Type of transition to the extension"
    )
    duration: Literal[8] = Field(8, description="Extension duration (always 8s for Veo)")

class VideoSequenceExtensionResponse(BaseModel):
    """Response for video sequence extension."""
    original_video_path: str = Field(description="Original video path")
    operation_name: str = Field(description="Extension operation ID")
    extension_concept: str = Field(description="Extension concept applied")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    generation_prompt: str = Field(description="Prompt used for extension")

class ImageToVideoRequest(BaseModel):
    """Request for converting images to video."""
    image_path: str = Field(description="Path to the source image")
    animation_concept: str = Field(description="How to animate the image")
    style: Literal["cinematic", "artistic", "realistic", "abstract"] = Field(
        "cinematic", description="Animation style"
    )
    duration: Literal[4, 6, 8] = Field(8, description="Video duration in seconds")
    aspect_ratio: Literal["16:9", "9:16"] = Field("16:9", description="Video aspect ratio")
    movement_intensity: Literal["subtle", "moderate", "dramatic"] = Field(
        "moderate", description="Intensity of movement/animation"
    )
    preserve_composition: bool = Field(True, description="Preserve the original image composition")

class ImageToVideoResponse(BaseModel):
    """Response for image to video conversion."""
    source_image_path: str = Field(description="Source image path")
    operation_name: str = Field(description="Video generation operation ID")
    animation_concept: str = Field(description="Animation concept applied")
    style_applied: str = Field(description="Style applied")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    generation_prompt: str = Field(description="Prompt used for animation")

class FrameDirectedVideoRequest(BaseModel):
    """Request for frame-directed video generation."""
    first_frame_path: str = Field(description="Path to the first frame image")
    last_frame_path: str = Field(description="Path to the last frame image")
    transition_concept: str = Field(description="How to transition between frames")
    style: Literal["cinematic", "artistic", "smooth", "dramatic"] = Field(
        "cinematic", description="Transition style"
    )
    duration: Literal[8] = Field(8, description="Video duration (always 8s for frame-directed)")
    aspect_ratio: Literal["16:9", "9:16"] = Field("16:9", description="Video aspect ratio")
    maintain_subjects: bool = Field(True, description="Maintain subjects from both frames")

class FrameDirectedVideoResponse(BaseModel):
    """Response for frame-directed video generation."""
    first_frame_path: str = Field(description="First frame path")
    last_frame_path: str = Field(description="Last frame path")
    operation_name: str = Field(description="Video generation operation ID")
    transition_concept: str = Field(description="Transition concept applied")
    style_applied: str = Field(description="Style applied")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    generation_prompt: str = Field(description="Prompt used for generation")

class VideoProductionWorkflowRequest(BaseModel):
    """Request for complete video production workflow."""
    artist_name: str = Field(description="Artist name")
    song_title: str = Field(description="Song title")
    concepts: List[str] = Field(description="List of visual concepts for different segments")
    workflow_type: Literal["sequential", "parallel", "mixed"] = Field(
        "sequential", description="How to process the concepts"
    )
    total_duration_target: int = Field(24, description="Target total duration in seconds")
    style_consistency: bool = Field(True, description="Maintain consistent style across segments")
    transition_style: Literal["cuts", "fades", "creative"] = Field(
        "cuts", description="Style of transitions between segments"
    )

class VideoProductionWorkflowResponse(BaseModel):
    """Response for video production workflow."""
    artist_name: str = Field(description="Artist name")
    song_title: str = Field(description="Song title")
    workflow_id: str = Field(description="Unique workflow identifier")
    segment_operations: List[str] = Field(description="List of operation IDs for each segment")
    estimated_total_time: Optional[str] = Field(None, description="Estimated total completion time")
    workflow_status: Literal["initiated", "processing", "completed", "failed"] = Field(
        "initiated", description="Overall workflow status"
    )
    segments_info: List[Dict[str, Any]] = Field(description="Information about each segment")

class VideoSaveRequest(BaseModel):
    """Request for saving generated video."""
    operation_name: str = Field(description="Video generation operation ID")
    output_path: str = Field(description="Path where to save the video")
    wait_for_completion: bool = Field(True, description="Whether to wait for generation completion")
    max_wait_time: int = Field(600, description="Maximum wait time in seconds")

class VideoSaveResponse(BaseModel):
    """Response for video save operation."""
    operation_name: str = Field(description="Video generation operation ID")
    output_path: str = Field(description="Path where video was saved")
    video_info: Optional[Dict[str, Any]] = Field(None, description="Video metadata")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    success: bool = Field(description="Whether save was successful")

# MCP Context for persistent client connections
class VideoProductionContext:
    """Context for video production client."""
    def __init__(self, veo_client: VeoVideoClient):
        self.veo_client = veo_client
        self.active_operations: Dict[str, Dict[str, Any]] = {}  # Track operations
        self.workflow_operations: Dict[str, List[str]] = {}  # Track workflows

# Lifespan context manager for API clients
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[Any]:
    """Initialize and manage API client connections."""
    try:
        # Initialize Veo client
        veo_client = VeoVideoClient()
        
        # Enter async context manager
        await veo_client.__aenter__()
        
        logger.info("Video production client initialized successfully")
        
        yield VideoProductionContext(veo_client=veo_client)
        
    except Exception as e:
        logger.error(f"Failed to initialize video production client: {str(e)}")
        raise ValueError(f"Failed to initialize video production client: {str(e)}")
    finally:
        # Clean up client connection
        try:
            await veo_client.__aexit__(None, None, None)
            logger.info("Video production client closed successfully")
        except Exception as e:
            logger.error(f"Error closing client: {str(e)}")

# Initialize FastMCP server
mcp = FastMCP("Video Production Server", lifespan=lifespan)

# MCP Tools

@mcp.tool()
async def video_generate_music_video_clip(request: MusicVideoClipRequest, ctx: Context) -> MusicVideoClipResponse:
    """Generate high-quality music video clips using AI video generation.
    
    Creates professional music video content with specified artistic style,
    mood, and visual concepts. Optimized for music video production workflows.
    """
    try:
        await ctx.info(f"Generating music video clip for '{request.song_title}' by {request.artist_name}")
        
        clients = ctx.request_context.lifespan_context
        
        # Build comprehensive video prompt
        prompt_parts = [
            f"Music video for '{request.song_title}' by {request.artist_name}"
        ]
        
        # Add concept description
        prompt_parts.append(request.concept_description)
        
        # Add style descriptors
        style_descriptors = {
            "cinematic": "cinematic quality, professional cinematography, dramatic lighting, film-like composition",
            "artistic": "artistic interpretation, creative visuals, expressive imagery, fine art aesthetic",
            "performance": "live performance energy, stage presence, concert atmosphere, musical performance",
            "narrative": "storytelling elements, narrative structure, character-driven scenes, plot development",
            "abstract": "abstract visuals, creative interpretation, non-literal imagery, artistic abstraction",
            "retro": "vintage aesthetic, retro styling, nostalgic atmosphere, classic music video feel"
        }
        
        prompt_parts.append(style_descriptors.get(request.style, ""))
        
        # Add mood and atmosphere
        if request.mood:
            prompt_parts.append(f"{request.mood} mood and emotional tone")
        
        # Add performance elements
        if request.include_performance:
            prompt_parts.append("featuring musical performance elements, artist presence")
        
        # Add visual elements
        if request.color_palette:
            prompt_parts.append(f"{request.color_palette} color palette")
        
        if request.camera_style:
            prompt_parts.append(f"{request.camera_style} camera movement")
        
        # Add music synchronization
        if request.sync_to_music:
            prompt_parts.append("synchronized to musical rhythm and energy")
        
        # Add quality enhancers
        prompt_parts.extend([
            "high production value", "professional music video quality",
            "dynamic visual composition", "rich colors", "dramatic lighting",
            "award-winning cinematography"
        ])
        
        final_prompt = ", ".join(prompt_parts)
        
        # Configure video generation
        video_config = VideoGenerationConfig(
            resolution=request.resolution,
            aspect_ratio=request.aspect_ratio,
            duration=request.duration,
            person_generation=request.include_performance
        )
        
        # Create generation request
        gen_request = TextToVideoRequest(
            prompt=final_prompt,
            config=video_config
        )
        
        # Start video generation
        generation_response = await clients.veo_client.generate_text_to_video(gen_request)
        
        # Store operation info
        clients.active_operations[generation_response.operation_name] = {
            "type": "music_video_clip",
            "artist_name": request.artist_name,
            "song_title": request.song_title,
            "concept": request.concept_description,
            "style": request.style,
            "started_at": asyncio.get_event_loop().time()
        }
        
        response = MusicVideoClipResponse(
            artist_name=request.artist_name,
            song_title=request.song_title,
            operation_name=generation_response.operation_name,
            estimated_completion_time=generation_response.estimated_completion_time,
            concept_applied=request.concept_description,
            style_applied=request.style,
            generation_prompt=final_prompt,
            video_config={
                "duration": request.duration,
                "resolution": request.resolution,
                "aspect_ratio": request.aspect_ratio,
                "style": request.style
            }
        )
        
        await ctx.info(f"Video generation started. Operation ID: {generation_response.operation_name}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate music video clip: {str(e)}")
        raise ValueError(f"Failed to generate music video clip: {str(e)}")

@mcp.tool()
async def video_check_generation_status(request: VideoGenerationStatusRequest, ctx: Context) -> VideoGenerationStatusResponse:
    """Check the status of video generation operations.
    
    Monitors the progress of video generation and returns current status,
    including completion percentage and estimated time remaining.
    """
    try:
        await ctx.info(f"Checking status for operation: {request.operation_name}")
        
        clients = ctx.request_context.lifespan_context
        
        # Get operation status from Veo
        operation_status = await clients.veo_client._get_operation_status(request.operation_name)
        
        # Determine status
        if operation_status.done:
            if operation_status.error:
                status = "failed"
                error_message = str(operation_status.error)
                result = None
            else:
                status = "completed"
                error_message = None
                # Parse result if available
                if operation_status.response:
                    try:
                        result = clients.veo_client._parse_result(operation_status.response)
                    except Exception as e:
                        logger.error(f"Failed to parse result: {str(e)}")
                        result = None
                else:
                    result = None
        else:
            status = "running"
            error_message = None
            result = None
        
        # Calculate progress (rough estimate based on time)
        progress = None
        if request.operation_name in clients.active_operations:
            op_info = clients.active_operations[request.operation_name]
            elapsed_time = asyncio.get_event_loop().time() - op_info["started_at"]
            # Rough estimate: video generation typically takes 2-5 minutes
            estimated_total_time = 180  # 3 minutes average
            progress = min(95, (elapsed_time / estimated_total_time) * 100)
            
            if status == "completed":
                progress = 100
        
        response = VideoGenerationStatusResponse(
            operation_name=request.operation_name,
            status=status,
            progress=progress,
            estimated_completion_time=operation_status.metadata.get("estimatedCompletionTime") if operation_status.metadata else None,
            error_message=error_message,
            result=result
        )
        
        await ctx.info(f"Operation status: {status}" + (f" ({progress:.1f}% complete)" if progress else ""))
        return response
        
    except Exception as e:
        logger.error(f"Failed to check generation status: {str(e)}")
        raise ValueError(f"Failed to check generation status: {str(e)}")

@mcp.tool()
async def video_extend_sequence(request: VideoSequenceExtensionRequest, ctx: Context) -> VideoSequenceExtensionResponse:
    """Extend existing video sequences with additional content.
    
    Takes an existing video and creates a seamless extension with new content,
    maintaining visual consistency and style continuity.
    """
    try:
        await ctx.info(f"Extending video sequence: {request.video_path}")
        
        clients = ctx.request_context.lifespan_context
        
        # Prepare video input
        video_input = clients.veo_client._prepare_video_input(request.video_path)
        
        # Build extension prompt
        prompt_parts = [request.extension_concept]
        
        if request.maintain_style:
            prompt_parts.append("maintaining the visual style and aesthetic of the original video")
        
        # Add transition styling
        transition_descriptors = {
            "smooth": "smooth, seamless transition",
            "cut": "clean cut transition",
            "fade": "gradual fade transition",
            "creative": "creative, artistic transition"
        }
        
        prompt_parts.append(transition_descriptors.get(request.transition_type, ""))
        
        # Add quality enhancers
        prompt_parts.extend([
            "consistent visual quality", "professional cinematography",
            "seamless continuation", "high production value"
        ])
        
        final_prompt = ", ".join(prompt_parts)
        
        # Configure extension
        video_config = VideoGenerationConfig(
            duration=request.duration  # Always 8s for extensions
        )
        
        # Create extension request
        ext_request = VideoExtensionRequest(
            video=video_input,
            prompt=final_prompt,
            config=video_config
        )
        
        # Start extension generation
        extension_response = await clients.veo_client.extend_video(ext_request)
        
        # Store operation info
        clients.active_operations[extension_response.operation_name] = {
            "type": "video_extension",
            "original_video": request.video_path,
            "extension_concept": request.extension_concept,
            "started_at": asyncio.get_event_loop().time()
        }
        
        response = VideoSequenceExtensionResponse(
            original_video_path=request.video_path,
            operation_name=extension_response.operation_name,
            extension_concept=request.extension_concept,
            estimated_completion_time=extension_response.estimated_completion_time,
            generation_prompt=final_prompt
        )
        
        await ctx.info(f"Video extension started. Operation ID: {extension_response.operation_name}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to extend video sequence: {str(e)}")
        raise ValueError(f"Failed to extend video sequence: {str(e)}")

@mcp.tool()
async def video_animate_image(request: ImageToVideoRequest, ctx: Context) -> ImageToVideoResponse:
    """Convert static images into animated video sequences.
    
    Brings static images to life with AI-powered animation,
    perfect for creating dynamic content from album covers or promotional images.
    """
    try:
        await ctx.info(f"Animating image: {request.image_path}")
        
        clients = ctx.request_context.lifespan_context
        
        # Prepare image input
        image_input = clients.veo_client._prepare_image_input(request.image_path)
        
        # Build animation prompt
        prompt_parts = [request.animation_concept]
        
        # Add style descriptors
        style_descriptors = {
            "cinematic": "cinematic animation, film-like movement, professional cinematography",
            "artistic": "artistic animation, creative movement, expressive motion",
            "realistic": "realistic animation, natural movement, lifelike motion",
            "abstract": "abstract animation, creative interpretation, artistic movement"
        }
        
        prompt_parts.append(style_descriptors.get(request.style, ""))
        
        # Add movement intensity
        intensity_descriptors = {
            "subtle": "subtle, gentle movement",
            "moderate": "moderate, balanced movement",
            "dramatic": "dramatic, dynamic movement"
        }
        
        prompt_parts.append(intensity_descriptors.get(request.movement_intensity, ""))
        
        if request.preserve_composition:
            prompt_parts.append("preserving the original composition and main elements")
        
        # Add quality enhancers
        prompt_parts.extend([
            "smooth animation", "high quality motion",
            "professional animation", "seamless movement"
        ])
        
        final_prompt = ", ".join(prompt_parts)
        
        # Configure video generation
        video_config = VideoGenerationConfig(
            resolution="1080p",
            aspect_ratio=request.aspect_ratio,
            duration=request.duration
        )
        
        # Create image-to-video request
        img_to_vid_request = ImageToVideoRequest(
            image=image_input,
            prompt=final_prompt,
            config=video_config
        )
        
        # Start animation generation
        animation_response = await clients.veo_client.generate_image_to_video(img_to_vid_request)
        
        # Store operation info
        clients.active_operations[animation_response.operation_name] = {
            "type": "image_to_video",
            "source_image": request.image_path,
            "animation_concept": request.animation_concept,
            "style": request.style,
            "started_at": asyncio.get_event_loop().time()
        }
        
        response = ImageToVideoResponse(
            source_image_path=request.image_path,
            operation_name=animation_response.operation_name,
            animation_concept=request.animation_concept,
            style_applied=request.style,
            estimated_completion_time=animation_response.estimated_completion_time,
            generation_prompt=final_prompt
        )
        
        await ctx.info(f"Image animation started. Operation ID: {animation_response.operation_name}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to animate image: {str(e)}")
        raise ValueError(f"Failed to animate image: {str(e)}")

@mcp.tool()
async def video_generate_frame_directed(request: FrameDirectedVideoRequest, ctx: Context) -> FrameDirectedVideoResponse:
    """Generate videos with specific start and end frames.
    
    Creates smooth transitions between two specific images,
    perfect for creating narrative sequences or artistic transitions.
    """
    try:
        await ctx.info(f"Generating frame-directed video from {request.first_frame_path} to {request.last_frame_path}")
        
        clients = ctx.request_context.lifespan_context
        
        # Prepare image inputs
        first_frame = clients.veo_client._prepare_image_input(request.first_frame_path)
        last_frame = clients.veo_client._prepare_image_input(request.last_frame_path)
        
        # Build transition prompt
        prompt_parts = [request.transition_concept]
        
        # Add style descriptors
        style_descriptors = {
            "cinematic": "cinematic transition, film-like quality, professional cinematography",
            "artistic": "artistic transition, creative interpretation, expressive movement",
            "smooth": "smooth, seamless transition, fluid movement",
            "dramatic": "dramatic transition, dynamic movement, impactful change"
        }
        
        prompt_parts.append(style_descriptors.get(request.style, ""))
        
        if request.maintain_subjects:
            prompt_parts.append("maintaining the main subjects from both frames")
        
        # Add quality enhancers
        prompt_parts.extend([
            "smooth transition", "professional quality",
            "seamless morphing", "high-quality animation"
        ])
        
        final_prompt = ", ".join(prompt_parts)
        
        # Configure video generation
        video_config = VideoGenerationConfig(
            resolution="1080p",
            aspect_ratio=request.aspect_ratio,
            duration=request.duration  # Always 8s for frame-directed
        )
        
        # Create frame-directed request
        frame_request = FrameDirectedRequest(
            first_frame=first_frame,
            last_frame=last_frame,
            prompt=final_prompt,
            config=video_config
        )
        
        # Start generation
        generation_response = await clients.veo_client.generate_frame_directed(frame_request)
        
        # Store operation info
        clients.active_operations[generation_response.operation_name] = {
            "type": "frame_directed",
            "first_frame": request.first_frame_path,
            "last_frame": request.last_frame_path,
            "transition_concept": request.transition_concept,
            "style": request.style,
            "started_at": asyncio.get_event_loop().time()
        }
        
        response = FrameDirectedVideoResponse(
            first_frame_path=request.first_frame_path,
            last_frame_path=request.last_frame_path,
            operation_name=generation_response.operation_name,
            transition_concept=request.transition_concept,
            style_applied=request.style,
            estimated_completion_time=generation_response.estimated_completion_time,
            generation_prompt=final_prompt
        )
        
        await ctx.info(f"Frame-directed generation started. Operation ID: {generation_response.operation_name}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate frame-directed video: {str(e)}")
        raise ValueError(f"Failed to generate frame-directed video: {str(e)}")

@mcp.tool()
async def video_save_generated(request: VideoSaveRequest, ctx: Context) -> VideoSaveResponse:
    """Save completed generated video to specified location.
    
    Downloads and saves the generated video file, optionally waiting
    for generation completion if the operation is still in progress.
    """
    try:
        await ctx.info(f"Saving video from operation: {request.operation_name}")
        
        clients = ctx.request_context.lifespan_context
        
        # Wait for completion if requested
        if request.wait_for_completion:
            await ctx.info("Waiting for video generation to complete...")
            result = await clients.veo_client.wait_for_completion(
                request.operation_name,
                poll_interval=30,
                max_wait_time=request.max_wait_time
            )
        else:
            # Check if already completed
            status = await clients.veo_client._get_operation_status(request.operation_name)
            if not status.done:
                raise ValueError("Video generation is not yet complete")
            
            if status.error:
                raise ValueError(f"Video generation failed: {status.error}")
            
            result = clients.veo_client._parse_result(status.response)
        
        # Save the video
        clients.veo_client.save_video(result.video, request.output_path)
        
        # Get file info
        output_file = Path(request.output_path)
        file_size = output_file.stat().st_size if output_file.exists() else None
        
        # Extract video metadata
        video_info = {
            "mime_type": result.video.mime_type,
            "width": result.video.width,
            "height": result.video.height,
            "frame_rate": result.video.frame_rate
        }
        
        response = VideoSaveResponse(
            operation_name=request.operation_name,
            output_path=request.output_path,
            video_info=video_info,
            file_size=file_size,
            duration=result.video.duration,
            success=True
        )
        
        await ctx.info(f"Video saved successfully to {request.output_path}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        return VideoSaveResponse(
            operation_name=request.operation_name,
            output_path=request.output_path,
            video_info=None,
            file_size=None,
            duration=None,
            success=False
        )

@mcp.tool()
async def video_create_production_workflow(request: VideoProductionWorkflowRequest, ctx: Context) -> VideoProductionWorkflowResponse:
    """Create a complete video production workflow with multiple segments.
    
    Orchestrates the creation of multiple video segments that can be combined
    into a complete music video, managing the entire production pipeline.
    """
    try:
        await ctx.info(f"Creating video production workflow for '{request.song_title}' by {request.artist_name}")
        
        clients = ctx.request_context.lifespan_context
        
        # Generate unique workflow ID
        import uuid
        workflow_id = str(uuid.uuid4())[:8]
        
        # Calculate duration per segment
        segments_count = len(request.concepts)
        duration_per_segment = min(8, request.total_duration_target // segments_count)
        
        # Ensure we use valid durations (4, 6, or 8)
        if duration_per_segment <= 4:
            duration_per_segment = 4
        elif duration_per_segment <= 6:
            duration_per_segment = 6
        else:
            duration_per_segment = 8
        
        segment_operations = []
        segments_info = []
        
        # Create base style for consistency
        base_style = "cinematic" if request.style_consistency else None
        
        # Process concepts based on workflow type
        if request.workflow_type == "parallel":
            # Start all segments simultaneously
            for i, concept in enumerate(request.concepts):
                segment_prompt = f"Music video segment {i+1} for '{request.song_title}' by {request.artist_name}: {concept}"
                
                if base_style:
                    segment_prompt += f", {base_style} style"
                
                segment_prompt += ", high production value, professional cinematography"
                
                # Configure segment
                video_config = VideoGenerationConfig(
                    resolution="1080p",
                    aspect_ratio="16:9",
                    duration=duration_per_segment
                )
                
                # Create generation request
                gen_request = TextToVideoRequest(
                    prompt=segment_prompt,
                    config=video_config
                )
                
                # Start generation
                generation_response = await clients.veo_client.generate_text_to_video(gen_request)
                segment_operations.append(generation_response.operation_name)
                
                # Store segment info
                segments_info.append({
                    "segment_number": i + 1,
                    "concept": concept,
                    "operation_id": generation_response.operation_name,
                    "duration": duration_per_segment,
                    "style": base_style or "cinematic",
                    "prompt": segment_prompt
                })
                
                # Store operation info
                clients.active_operations[generation_response.operation_name] = {
                    "type": "workflow_segment",
                    "workflow_id": workflow_id,
                    "segment_number": i + 1,
                    "concept": concept,
                    "started_at": asyncio.get_event_loop().time()
                }
        
        else:  # Sequential or mixed (treat as sequential for now)
            # Start segments one by one (for demo - in practice might want to start first few)
            for i, concept in enumerate(request.concepts[:1]):  # Start first segment only
                segment_prompt = f"Music video segment {i+1} for '{request.song_title}' by {request.artist_name}: {concept}"
                
                if base_style:
                    segment_prompt += f", {base_style} style"
                
                segment_prompt += ", high production value, professional cinematography"
                
                # Configure segment
                video_config = VideoGenerationConfig(
                    resolution="1080p",
                    aspect_ratio="16:9",
                    duration=duration_per_segment
                )
                
                # Create generation request
                gen_request = TextToVideoRequest(
                    prompt=segment_prompt,
                    config=video_config
                )
                
                # Start generation
                generation_response = await clients.veo_client.generate_text_to_video(gen_request)
                segment_operations.append(generation_response.operation_name)
                
                # Store segment info
                segments_info.append({
                    "segment_number": i + 1,
                    "concept": concept,
                    "operation_id": generation_response.operation_name,
                    "duration": duration_per_segment,
                    "style": base_style or "cinematic",
                    "prompt": segment_prompt,
                    "status": "started"
                })
                
                # Store operation info
                clients.active_operations[generation_response.operation_name] = {
                    "type": "workflow_segment",
                    "workflow_id": workflow_id,
                    "segment_number": i + 1,
                    "concept": concept,
                    "started_at": asyncio.get_event_loop().time()
                }
            
            # Add remaining segments as pending
            for i, concept in enumerate(request.concepts[1:], start=1):
                segments_info.append({
                    "segment_number": i + 1,
                    "concept": concept,
                    "operation_id": None,
                    "duration": duration_per_segment,
                    "style": base_style or "cinematic",
                    "prompt": None,
                    "status": "pending"
                })
        
        # Store workflow info
        clients.workflow_operations[workflow_id] = segment_operations
        
        # Estimate total completion time
        if request.workflow_type == "parallel":
            estimated_minutes = 5  # All segments run in parallel
        else:
            estimated_minutes = len(request.concepts) * 3  # Sequential, ~3 min per segment
        
        estimated_total_time = f"~{estimated_minutes} minutes"
        
        response = VideoProductionWorkflowResponse(
            artist_name=request.artist_name,
            song_title=request.song_title,
            workflow_id=workflow_id,
            segment_operations=segment_operations,
            estimated_total_time=estimated_total_time,
            workflow_status="processing" if segment_operations else "initiated",
            segments_info=segments_info
        )
        
        await ctx.info(f"Workflow {workflow_id} created with {len(segments_info)} segments")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create production workflow: {str(e)}")
        raise ValueError(f"Failed to create production workflow: {str(e)}")

def main():
    """Run the Video Production MCP server."""
    logger.info("Starting Video Production MCP server")
    mcp.run()

if __name__ == "__main__":
    main()
