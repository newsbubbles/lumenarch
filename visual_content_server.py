"""Visual Content MCP Server for Empire.AI

Provides visual content creation and research capabilities through Gemini and Google Custom Search APIs.
Exposes tools for image generation, editing, and reference image discovery.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Any, Optional, List, Dict, Literal
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context

# Import our API clients
from clients.gemini.client import (
    GeminiImageClient, TextToImageRequest, ImageEditRequest, MultiImageRequest,
    ImageGenerationResponse, GeneratedImage, ImageInput, TextInput,
    GenerationConfig, SafetySetting
)
from clients.google_search.client import (
    GoogleCustomSearchClient, ImageSearchRequest, WebSearchRequest,
    SearchResponse, SearchImage
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
logger.info("Visual Content MCP server starting")

# Request/Response Models for MCP Tools

class ConceptImageRequest(BaseModel):
    """Request for generating concept art for music videos."""
    concept_description: str = Field(description="Detailed description of the visual concept")
    artist_name: Optional[str] = Field(None, description="Artist name for context")
    song_title: Optional[str] = Field(None, description="Song title for context")
    style: Literal["cinematic", "artistic", "photorealistic", "abstract", "vintage", "modern"] = Field(
        "cinematic", description="Visual style for the concept"
    )
    mood: Optional[str] = Field(None, description="Emotional mood (e.g., dark, upbeat, melancholic)")
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3"] = Field(
        "16:9", description="Image aspect ratio"
    )
    color_scheme: Optional[str] = Field(None, description="Preferred color scheme")
    include_text: bool = Field(False, description="Whether to include text elements")

class ConceptImageResponse(BaseModel):
    """Response for concept image generation."""
    concept_description: str = Field(description="Original concept description")
    generated_images: List[GeneratedImage] = Field(description="Generated concept images")
    style_applied: str = Field(description="Visual style that was applied")
    generation_metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions for variations")

class ReferenceImageSearchRequest(BaseModel):
    """Request for searching reference images."""
    search_query: str = Field(description="Search query for reference images")
    artist_name: Optional[str] = Field(None, description="Artist name to include in search")
    era: Optional[str] = Field(None, description="Time period (e.g., '1980s', '1990s', 'vintage')")
    image_type: Literal["photo", "face", "clipart", "lineart", "animated"] = Field(
        "photo", description="Type of images to search for"
    )
    color_type: Literal["color", "gray", "mono", "trans"] = Field(
        "color", description="Color preference"
    )
    size: Literal["icon", "small", "medium", "large", "xlarge", "xxlarge", "huge"] = Field(
        "large", description="Image size preference"
    )
    usage_rights: Literal["all", "labeled_for_reuse", "labeled_for_reuse_with_modification"] = Field(
        "all", description="Usage rights filter"
    )
    safe_search: Literal["off", "medium", "high"] = Field(
        "medium", description="Safe search level"
    )
    limit: int = Field(10, description="Maximum number of results")

class ReferenceImageSearchResponse(BaseModel):
    """Response for reference image search."""
    search_query: str = Field(description="Original search query")
    total_results: int = Field(description="Total number of results found")
    images: List[SearchImage] = Field(description="Found reference images")
    search_metadata: Optional[Dict[str, Any]] = Field(None, description="Search metadata")
    suggestions: Optional[List[str]] = Field(None, description="Related search suggestions")

class ArtistVisualRequest(BaseModel):
    """Request for generating artist-specific visual content."""
    artist_name: str = Field(description="Artist name")
    visual_type: Literal["portrait", "concept_art", "logo", "album_cover", "performance"] = Field(
        description="Type of visual content to create"
    )
    style_reference: Optional[str] = Field(None, description="Style reference or description")
    mood: Optional[str] = Field(None, description="Emotional tone or mood")
    era: Optional[str] = Field(None, description="Time period or era")
    color_palette: Optional[str] = Field(None, description="Color palette preference")
    include_instruments: bool = Field(False, description="Include musical instruments")
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3"] = Field(
        "1:1", description="Image aspect ratio"
    )

class ArtistVisualResponse(BaseModel):
    """Response for artist visual generation."""
    artist_name: str = Field(description="Artist name")
    visual_type: str = Field(description="Type of visual created")
    generated_images: List[GeneratedImage] = Field(description="Generated artist visuals")
    style_applied: str = Field(description="Style that was applied")
    generation_prompt: str = Field(description="Final prompt used for generation")
    variations: Optional[List[str]] = Field(None, description="Suggested variations")

class ImageEditRequest(BaseModel):
    """Request for editing existing images."""
    image_path: str = Field(description="Path to the image to edit")
    edit_instruction: str = Field(description="Description of edits to make")
    style_transfer: Optional[str] = Field(None, description="Style to transfer to the image")
    mood_adjustment: Optional[str] = Field(None, description="Mood adjustment (brighter, darker, warmer, etc.)")
    aspect_ratio: Optional[Literal["1:1", "16:9", "9:16", "4:3"]] = Field(
        None, description="New aspect ratio (if resizing)"
    )
    preserve_subject: bool = Field(True, description="Whether to preserve the main subject")

class ImageEditResponse(BaseModel):
    """Response for image editing."""
    original_image_path: str = Field(description="Path to original image")
    edited_images: List[GeneratedImage] = Field(description="Edited image results")
    edit_instruction: str = Field(description="Edit instruction that was applied")
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")

class MultiImageCompositionRequest(BaseModel):
    """Request for combining multiple images into a composition."""
    image_paths: List[str] = Field(description="Paths to images to combine")
    composition_style: Literal["collage", "blend", "montage", "grid", "artistic"] = Field(
        "artistic", description="Style of composition"
    )
    layout_instruction: str = Field(description="How to arrange/combine the images")
    theme: Optional[str] = Field(None, description="Overall theme for the composition")
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3"] = Field(
        "16:9", description="Final composition aspect ratio"
    )
    include_text: bool = Field(False, description="Whether to include text elements")

class MultiImageCompositionResponse(BaseModel):
    """Response for multi-image composition."""
    source_images: List[str] = Field(description="Source image paths")
    composition: GeneratedImage = Field(description="Final composed image")
    composition_style: str = Field(description="Style used for composition")
    layout_description: str = Field(description="Description of the final layout")

class VisualResearchRequest(BaseModel):
    """Request for comprehensive visual research."""
    research_topic: str = Field(description="Topic to research visually")
    artist_name: Optional[str] = Field(None, description="Artist name for context")
    time_period: Optional[str] = Field(None, description="Specific time period")
    visual_categories: List[Literal["photos", "artwork", "performances", "fashion", "venues", "instruments"]] = Field(
        ["photos"], description="Categories of visuals to research"
    )
    include_cultural_context: bool = Field(True, description="Include cultural/historical context")
    max_results_per_category: int = Field(5, description="Maximum results per category")

class VisualResearchResponse(BaseModel):
    """Response for visual research."""
    research_topic: str = Field(description="Research topic")
    visual_categories: Dict[str, List[SearchImage]] = Field(description="Categorized visual results")
    cultural_context: Optional[List[str]] = Field(None, description="Cultural/historical insights")
    visual_themes: Optional[List[str]] = Field(None, description="Common visual themes found")
    inspiration_suggestions: Optional[List[str]] = Field(None, description="Creative inspiration suggestions")
    total_results: int = Field(description="Total number of visual results")

# MCP Context for persistent client connections
class VisualContentContext:
    """Context for visual content clients."""
    def __init__(self, gemini_client: GeminiImageClient, search_client: GoogleCustomSearchClient):
        self.gemini_client = gemini_client
        self.search_client = search_client

# Lifespan context manager for API clients
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[Any]:
    """Initialize and manage API client connections."""
    try:
        # Initialize clients
        gemini_client = GeminiImageClient()
        search_client = GoogleCustomSearchClient()
        
        # Enter async context managers
        await gemini_client.__aenter__()
        await search_client.__aenter__()
        
        logger.info("Visual content clients initialized successfully")
        
        yield VisualContentContext(
            gemini_client=gemini_client,
            search_client=search_client
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize visual content clients: {str(e)}")
        raise ValueError(f"Failed to initialize visual content clients: {str(e)}")
    finally:
        # Clean up client connections
        try:
            await gemini_client.__aexit__(None, None, None)
            await search_client.__aexit__(None, None, None)
            logger.info("Visual content clients closed successfully")
        except Exception as e:
            logger.error(f"Error closing clients: {str(e)}")

# Initialize FastMCP server
mcp = FastMCP("Visual Content Server", lifespan=lifespan)

# MCP Tools

@mcp.tool()
async def visual_generate_concept_image(request: ConceptImageRequest, ctx: Context) -> ConceptImageResponse:
    """Generate concept art for music videos using AI image generation.
    
    Creates original visual concepts based on detailed descriptions, artist context,
    and specified visual styles. Perfect for brainstorming and initial concept development.
    """
    try:
        await ctx.info(f"Generating concept image: {request.concept_description[:50]}...")
        
        clients = ctx.request_context.lifespan_context
        
        # Build comprehensive prompt
        prompt_parts = [request.concept_description]
        
        if request.artist_name:
            prompt_parts.append(f"in the style associated with {request.artist_name}")
        
        if request.song_title:
            prompt_parts.append(f"inspired by the song '{request.song_title}'")
        
        # Add style descriptors
        style_descriptors = {
            "cinematic": "cinematic lighting, dramatic composition, film-like quality, professional cinematography",
            "artistic": "artistic interpretation, creative composition, expressive style, fine art quality",
            "photorealistic": "photorealistic, high detail, realistic lighting, professional photography",
            "abstract": "abstract interpretation, creative shapes, artistic abstraction, non-literal representation",
            "vintage": "vintage aesthetic, retro styling, aged look, classic composition",
            "modern": "modern aesthetic, contemporary styling, clean lines, current trends"
        }
        
        prompt_parts.append(style_descriptors.get(request.style, ""))
        
        if request.mood:
            prompt_parts.append(f"{request.mood} mood and atmosphere")
        
        if request.color_scheme:
            prompt_parts.append(f"{request.color_scheme} color palette")
        
        # Add quality enhancers
        prompt_parts.extend([
            "high quality", "detailed", "professional", "8K resolution",
            "award-winning composition"
        ])
        
        final_prompt = ", ".join(prompt_parts)
        
        # Configure generation settings
        config = GenerationConfig(
            aspect_ratio=request.aspect_ratio,
            style=request.style
        )
        
        # Create generation request
        gen_request = TextToImageRequest(
            prompt=final_prompt,
            config=config
        )
        
        # Generate image
        generation_response = await clients.gemini_client.generate_text_to_image(gen_request)
        
        # Create suggestions for variations
        suggestions = [
            f"Try with different lighting: dramatic shadows, soft lighting, neon lighting",
            f"Experiment with camera angles: low angle, bird's eye view, close-up",
            f"Explore color variations: monochrome, warm tones, cool tones",
            f"Add environmental elements: urban setting, nature, studio setup"
        ]
        
        response = ConceptImageResponse(
            concept_description=request.concept_description,
            generated_images=generation_response.images,
            style_applied=request.style,
            generation_metadata={
                "final_prompt": final_prompt,
                "aspect_ratio": request.aspect_ratio,
                "artist_context": request.artist_name,
                "song_context": request.song_title
            },
            suggestions=suggestions
        )
        
        await ctx.info(f"Generated {len(generation_response.images)} concept images")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate concept image: {str(e)}")
        raise ValueError(f"Failed to generate concept image: {str(e)}")

@mcp.tool()
async def visual_search_reference_images(request: ReferenceImageSearchRequest, ctx: Context) -> ReferenceImageSearchResponse:
    """Search for reference images to inspire visual concepts.
    
    Finds relevant images based on artist, era, style, or concept keywords.
    Useful for gathering visual inspiration and understanding aesthetic trends.
    """
    try:
        await ctx.info(f"Searching for reference images: {request.search_query}")
        
        clients = ctx.request_context.lifespan_context
        
        # Build comprehensive search query
        search_terms = [request.search_query]
        
        if request.artist_name:
            search_terms.append(request.artist_name)
        
        if request.era:
            search_terms.append(request.era)
        
        final_query = " ".join(search_terms)
        
        # Create search request
        search_request = ImageSearchRequest(
            query=final_query,
            num_results=request.limit,
            img_type=request.image_type,
            img_color_type=request.color_type,
            img_size=request.size,
            rights=request.usage_rights,
            safe=request.safe_search
        )
        
        # Perform search
        search_results = await clients.search_client.search_images(search_request)
        
        # Generate related suggestions
        suggestions = []
        if request.artist_name:
            suggestions.extend([
                f"{request.artist_name} concert photos",
                f"{request.artist_name} album covers",
                f"{request.artist_name} vintage photos"
            ])
        
        if request.era:
            suggestions.extend([
                f"{request.era} fashion",
                f"{request.era} photography",
                f"{request.era} aesthetic"
            ])
        
        suggestions.extend([
            f"{request.search_query} artistic",
            f"{request.search_query} professional",
            f"{request.search_query} vintage"
        ])
        
        response = ReferenceImageSearchResponse(
            search_query=request.search_query,
            total_results=len(search_results.images),
            images=search_results.images,
            search_metadata={
                "final_query": final_query,
                "image_type": request.image_type,
                "color_type": request.color_type,
                "size": request.size,
                "usage_rights": request.usage_rights
            },
            suggestions=suggestions[:10]  # Limit suggestions
        )
        
        await ctx.info(f"Found {len(search_results.images)} reference images")
        return response
        
    except Exception as e:
        logger.error(f"Failed to search reference images: {str(e)}")
        raise ValueError(f"Failed to search reference images: {str(e)}")

@mcp.tool()
async def visual_create_artist_visual(request: ArtistVisualRequest, ctx: Context) -> ArtistVisualResponse:
    """Generate artist-specific visual content like portraits, logos, or concept art.
    
    Creates customized visuals tailored to specific artists, incorporating their
    style, era, and musical characteristics into the visual design.
    """
    try:
        await ctx.info(f"Creating {request.visual_type} for artist: {request.artist_name}")
        
        clients = ctx.request_context.lifespan_context
        
        # Build artist-specific prompt based on visual type
        prompt_parts = []
        
        if request.visual_type == "portrait":
            prompt_parts.extend([
                f"Professional portrait of {request.artist_name}",
                "artistic photography", "professional lighting",
                "expressive pose", "music artist portrait"
            ])
        elif request.visual_type == "concept_art":
            prompt_parts.extend([
                f"Concept art inspired by {request.artist_name}",
                "artistic interpretation", "creative visualization",
                "music-inspired artwork", "conceptual design"
            ])
        elif request.visual_type == "logo":
            prompt_parts.extend([
                f"Logo design for {request.artist_name}",
                "clean typography", "music branding",
                "professional logo", "artist identity"
            ])
        elif request.visual_type == "album_cover":
            prompt_parts.extend([
                f"Album cover design for {request.artist_name}",
                "music album artwork", "creative composition",
                "professional design", "artistic layout"
            ])
        elif request.visual_type == "performance":
            prompt_parts.extend([
                f"{request.artist_name} live performance",
                "concert photography", "stage lighting",
                "dynamic performance", "live music energy"
            ])
        
        # Add style and mood elements
        if request.style_reference:
            prompt_parts.append(f"in the style of {request.style_reference}")
        
        if request.mood:
            prompt_parts.append(f"{request.mood} mood and atmosphere")
        
        if request.era:
            prompt_parts.append(f"{request.era} aesthetic")
        
        if request.color_palette:
            prompt_parts.append(f"{request.color_palette} color scheme")
        
        if request.include_instruments:
            prompt_parts.append("featuring musical instruments")
        
        # Add quality enhancers
        prompt_parts.extend([
            "high quality", "professional", "detailed",
            "award-winning design", "artistic excellence"
        ])
        
        final_prompt = ", ".join(prompt_parts)
        
        # Configure generation
        config = GenerationConfig(
            aspect_ratio=request.aspect_ratio,
            style="artistic" if request.visual_type in ["concept_art", "logo"] else "photorealistic"
        )
        
        # Create generation request
        gen_request = TextToImageRequest(
            prompt=final_prompt,
            config=config
        )
        
        # Generate image
        generation_response = await clients.gemini_client.generate_text_to_image(gen_request)
        
        # Create variations suggestions
        variations = []
        if request.visual_type == "portrait":
            variations.extend([
                "Try different poses: profile, three-quarter view, action shot",
                "Experiment with lighting: dramatic shadows, soft light, colored gels",
                "Add environmental context: studio, stage, urban setting"
            ])
        elif request.visual_type == "logo":
            variations.extend([
                "Try different typography styles: modern, vintage, handwritten",
                "Experiment with symbols: musical notes, instruments, abstract shapes",
                "Explore color variations: monochrome, gradient, metallic"
            ])
        
        response = ArtistVisualResponse(
            artist_name=request.artist_name,
            visual_type=request.visual_type,
            generated_images=generation_response.images,
            style_applied=config.style,
            generation_prompt=final_prompt,
            variations=variations
        )
        
        await ctx.info(f"Generated {len(generation_response.images)} artist visuals")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create artist visual: {str(e)}")
        raise ValueError(f"Failed to create artist visual: {str(e)}")

@mcp.tool()
async def visual_edit_image(request: ImageEditRequest, ctx: Context) -> ImageEditResponse:
    """Edit existing images with AI-powered modifications.
    
    Applies edits, style transfers, mood adjustments, or other modifications
    to existing images while preserving or modifying key elements as requested.
    """
    try:
        await ctx.info(f"Editing image: {request.image_path}")
        
        clients = ctx.request_context.lifespan_context
        
        # Prepare image input
        image_input = clients.gemini_client._prepare_image_input(request.image_path)
        
        # Build edit instruction
        edit_parts = [request.edit_instruction]
        
        if request.style_transfer:
            edit_parts.append(f"Apply {request.style_transfer} style")
        
        if request.mood_adjustment:
            edit_parts.append(f"Adjust mood to be {request.mood_adjustment}")
        
        if request.preserve_subject:
            edit_parts.append("Preserve the main subject and composition")
        
        final_instruction = ", ".join(edit_parts)
        
        # Configure generation
        config = GenerationConfig()
        if request.aspect_ratio:
            config.aspect_ratio = request.aspect_ratio
        
        # Create edit request
        edit_req = ImageEditRequest(
            image=image_input,
            instruction=final_instruction,
            config=config
        )
        
        # Perform edit
        edit_response = await clients.gemini_client.edit_image(edit_req)
        
        response = ImageEditResponse(
            original_image_path=request.image_path,
            edited_images=edit_response.images,
            edit_instruction=final_instruction,
            processing_metadata={
                "style_transfer": request.style_transfer,
                "mood_adjustment": request.mood_adjustment,
                "preserve_subject": request.preserve_subject,
                "aspect_ratio": request.aspect_ratio
            }
        )
        
        await ctx.info(f"Generated {len(edit_response.images)} edited versions")
        return response
        
    except Exception as e:
        logger.error(f"Failed to edit image: {str(e)}")
        raise ValueError(f"Failed to edit image: {str(e)}")

@mcp.tool()
async def visual_compose_multi_images(request: MultiImageCompositionRequest, ctx: Context) -> MultiImageCompositionResponse:
    """Combine multiple images into artistic compositions.
    
    Creates cohesive visual compositions from multiple source images,
    useful for creating collages, mood boards, or complex visual narratives.
    """
    try:
        await ctx.info(f"Composing {len(request.image_paths)} images into {request.composition_style}")
        
        clients = ctx.request_context.lifespan_context
        
        # Prepare image inputs
        image_inputs = []
        for image_path in request.image_paths:
            image_input = clients.gemini_client._prepare_image_input(image_path)
            image_inputs.append(image_input)
        
        # Build composition instruction
        instruction_parts = [
            f"Create a {request.composition_style} composition",
            request.layout_instruction
        ]
        
        if request.theme:
            instruction_parts.append(f"with a {request.theme} theme")
        
        if request.include_text:
            instruction_parts.append("include artistic text elements")
        
        instruction_parts.extend([
            "maintain visual harmony",
            "professional composition",
            "artistic layout"
        ])
        
        final_instruction = ", ".join(instruction_parts)
        
        # Configure generation
        config = GenerationConfig(
            aspect_ratio=request.aspect_ratio,
            style="artistic"
        )
        
        # Create composition request
        comp_request = MultiImageRequest(
            images=image_inputs,
            instruction=final_instruction,
            config=config
        )
        
        # Perform composition
        comp_response = await clients.gemini_client.compose_images(comp_request)
        
        # Get the first (main) composed image
        composed_image = comp_response.images[0] if comp_response.images else None
        
        if not composed_image:
            raise ValueError("No composed image was generated")
        
        response = MultiImageCompositionResponse(
            source_images=request.image_paths,
            composition=composed_image,
            composition_style=request.composition_style,
            layout_description=final_instruction
        )
        
        await ctx.info("Multi-image composition completed")
        return response
        
    except Exception as e:
        logger.error(f"Failed to compose images: {str(e)}")
        raise ValueError(f"Failed to compose images: {str(e)}")

@mcp.tool()
async def visual_research_comprehensive(request: VisualResearchRequest, ctx: Context) -> VisualResearchResponse:
    """Conduct comprehensive visual research across multiple categories.
    
    Performs systematic visual research to gather inspiration, understand trends,
    and collect reference materials for creative projects.
    """
    try:
        await ctx.info(f"Conducting visual research on: {request.research_topic}")
        
        clients = ctx.request_context.lifespan_context
        visual_categories = {}
        total_results = 0
        
        # Search each requested category
        for category in request.visual_categories:
            try:
                # Build category-specific search query
                search_terms = [request.research_topic]
                
                category_keywords = {
                    "photos": ["photography", "photos"],
                    "artwork": ["art", "painting", "illustration"],
                    "performances": ["live", "concert", "performance"],
                    "fashion": ["style", "fashion", "outfit"],
                    "venues": ["venue", "stage", "location"],
                    "instruments": ["instruments", "equipment", "gear"]
                }
                
                if category in category_keywords:
                    search_terms.extend(category_keywords[category])
                
                if request.artist_name:
                    search_terms.append(request.artist_name)
                
                if request.time_period:
                    search_terms.append(request.time_period)
                
                final_query = " ".join(search_terms)
                
                # Create search request
                search_request = ImageSearchRequest(
                    query=final_query,
                    num_results=request.max_results_per_category,
                    img_type="photo" if category in ["photos", "performances", "venues"] else "clipart",
                    img_size="large"
                )
                
                # Perform search
                search_results = await clients.search_client.search_images(search_request)
                visual_categories[category] = search_results.images
                total_results += len(search_results.images)
                
                await ctx.info(f"Found {len(search_results.images)} images for {category}")
                
            except Exception as e:
                logger.error(f"Failed to search {category}: {str(e)}")
                visual_categories[category] = []
        
        # Analyze visual themes (simple keyword analysis)
        visual_themes = []
        if request.include_cultural_context:
            # Add time period context
            if request.time_period:
                if "1960s" in request.time_period.lower():
                    visual_themes.extend(["psychedelic", "counterculture", "folk revival"])
                elif "1970s" in request.time_period.lower():
                    visual_themes.extend(["disco", "glam rock", "punk emergence"])
                elif "1980s" in request.time_period.lower():
                    visual_themes.extend(["neon", "synth-pop", "MTV culture"])
                elif "1990s" in request.time_period.lower():
                    visual_themes.extend(["grunge", "alternative", "hip-hop culture"])
                elif "2000s" in request.time_period.lower():
                    visual_themes.extend(["digital", "emo", "pop-punk"])
        
        # Generate inspiration suggestions
        inspiration_suggestions = [
            f"Explore {request.research_topic} across different eras",
            f"Look for recurring visual motifs in {request.research_topic}",
            f"Consider the evolution of {request.research_topic} aesthetics",
            "Analyze color palettes and lighting techniques",
            "Study composition and framing approaches"
        ]
        
        if request.artist_name:
            inspiration_suggestions.extend([
                f"Compare {request.artist_name}'s visual evolution",
                f"Analyze {request.artist_name}'s signature visual elements"
            ])
        
        # Add cultural context insights
        cultural_context = []
        if request.include_cultural_context:
            cultural_context.extend([
                f"Visual research on {request.research_topic} reveals cultural trends",
                "Consider the historical context of visual elements",
                "Analyze the relationship between music and visual aesthetics"
            ])
            
            if request.time_period:
                cultural_context.append(f"The {request.time_period} era shows distinct visual characteristics")
        
        response = VisualResearchResponse(
            research_topic=request.research_topic,
            visual_categories=visual_categories,
            cultural_context=cultural_context if cultural_context else None,
            visual_themes=visual_themes if visual_themes else None,
            inspiration_suggestions=inspiration_suggestions,
            total_results=total_results
        )
        
        await ctx.info(f"Visual research completed. Total results: {total_results}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to conduct visual research: {str(e)}")
        raise ValueError(f"Failed to conduct visual research: {str(e)}")

def main():
    """Run the Visual Content MCP server."""
    logger.info("Starting Visual Content MCP server")
    mcp.run()

if __name__ == "__main__":
    main()
