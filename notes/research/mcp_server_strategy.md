# MCP Server Strategy - Empire.AI

*Created: 2025-11-14*

## Overview

This document outlines the strategy for wrapping the Empire.AI API clients into Model Context Protocol (MCP) servers. The goal is to create specialized MCP servers that expose the API functionality as tools for AI agents to orchestrate complex music video creation workflows.

## MCP Server Architecture

### Server Division Strategy

Based on functional domains and usage patterns, we'll create **three specialized MCP servers**:

#### 1. Music Research Server (`music_research_mcp.py`)
**Purpose**: Music metadata, artist research, and cultural context
**Wrapped Clients**:
- Spotify Client (music metadata, audio analysis)
- Last.fm Client (artist biographies, tags)
- Genius Client (lyrics, annotations)

**Key Tools**:
- `search_music` - Search for artists, albums, tracks across platforms
- `get_artist_profile` - Comprehensive artist information
- `analyze_audio_features` - Musical characteristics analysis
- `get_song_lyrics` - Lyrics and cultural context
- `find_similar_artists` - Artist discovery and recommendations

#### 2. Visual Content Server (`visual_content_mcp.py`) 
**Purpose**: Image generation, search, and visual concept development
**Wrapped Clients**:
- Gemini Image Client (AI image generation)
- Google Custom Search Client (reference image search)

**Key Tools**:
- `generate_concept_image` - Create original visual concepts
- `edit_image` - Modify and enhance existing images
- `search_reference_images` - Find inspiration and reference materials
- `create_artist_visual` - Generate artist-specific imagery
- `combine_images` - Multi-image composition

#### 3. Video Production Server (`video_production_mcp.py`)
**Purpose**: Video generation and music video creation
**Wrapped Clients**:
- Veo Video Client (AI video generation)

**Key Tools**:
- `generate_music_video_clip` - Create video sequences
- `extend_video_sequence` - Extend existing videos
- `create_frame_directed_video` - Generate with specific start/end frames
- `image_to_video` - Animate static images
- `check_video_status` - Monitor generation progress

## Tool Design Patterns

### Request/Response Mapping
Each MCP tool maps directly to client methods:

```python
# Client method
async def search(self, request: SearchRequest) -> SearchResponse:
    # Implementation

# MCP tool
@mcp_tool("search_music")
async def search_music_tool(
    query: str,
    search_types: List[str] = ["artist", "track"],
    limit: int = 20
) -> Dict[str, Any]:
    request = SearchRequest(
        query=query,
        search_type=search_types,
        limit=limit
    )
    response = await spotify_client.search(request)
    return response.model_dump()
```

### Error Handling Strategy
```python
@mcp_tool("tool_name")
async def tool_function(...):
    try:
        # Client operation
        result = await client.operation(request)
        return {
            "success": True,
            "data": result.model_dump(),
            "metadata": {"timestamp": datetime.now().isoformat()}
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
```

### Resource Management
```python
class MusicResearchServer:
    def __init__(self):
        self.spotify_client = None
        self.lastfm_client = None
        self.genius_client = None
    
    async def __aenter__(self):
        self.spotify_client = await SpotifyClient().__aenter__()
        self.lastfm_client = await LastFmClient().__aenter__()
        self.genius_client = await GeniusClient().__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.spotify_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.lastfm_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.genius_client.__aexit__(exc_type, exc_val, exc_tb)
```

## Tool Specifications

### Music Research Server Tools

#### `search_music`
**Description**: Search for music content across Spotify, Last.fm, and Genius
**Parameters**:
- `query` (string, required): Search query
- `platforms` (array, optional): Platforms to search ["spotify", "lastfm", "genius"]
- `content_types` (array, optional): Content types ["artist", "album", "track"]
- `limit` (integer, optional): Results per platform (default: 10)

**Returns**: Unified search results with platform attribution

#### `get_artist_profile`
**Description**: Get comprehensive artist information
**Parameters**:
- `artist_name` (string, required): Artist name
- `include_audio_analysis` (boolean, optional): Include Spotify audio features
- `include_biography` (boolean, optional): Include Last.fm biography
- `include_similar_artists` (boolean, optional): Include related artists

**Returns**: Complete artist profile with metadata

#### `analyze_song_context`
**Description**: Deep analysis of song lyrics and meaning
**Parameters**:
- `song_title` (string, required): Song title
- `artist_name` (string, required): Artist name
- `include_annotations` (boolean, optional): Include Genius annotations

**Returns**: Lyrics, annotations, and cultural context

### Visual Content Server Tools

#### `generate_concept_image`
**Description**: Generate original concept art for music videos
**Parameters**:
- `concept_description` (string, required): Visual concept description
- `artist_name` (string, optional): Artist for context
- `song_title` (string, optional): Song for context
- `style` (string, optional): Visual style ("cinematic", "artistic", etc.)
- `aspect_ratio` (string, optional): Image aspect ratio

**Returns**: Generated image data and metadata

#### `search_reference_images`
**Description**: Find reference images for visual concepts
**Parameters**:
- `search_query` (string, required): Image search query
- `image_type` (string, optional): Type filter ("photo", "clipart", etc.)
- `era` (string, optional): Time period filter
- `color_scheme` (string, optional): Color preference
- `limit` (integer, optional): Number of results

**Returns**: Curated reference image collection

#### `create_artist_visual`
**Description**: Generate artist-specific visual content
**Parameters**:
- `artist_name` (string, required): Artist name
- `visual_type` (string, required): Type ("portrait", "concept", "logo")
- `style_reference` (string, optional): Style description
- `mood` (string, optional): Emotional tone

**Returns**: Artist-specific generated image

### Video Production Server Tools

#### `generate_music_video_clip`
**Description**: Create music video sequences
**Parameters**:
- `concept_description` (string, required): Video concept
- `artist_name` (string, required): Artist name
- `song_title` (string, required): Song title
- `duration` (integer, optional): Video length in seconds (4-8)
- `style` (string, optional): Visual style
- `aspect_ratio` (string, optional): Video aspect ratio

**Returns**: Video generation job ID and estimated completion time

#### `check_video_status`
**Description**: Monitor video generation progress
**Parameters**:
- `job_id` (string, required): Video generation job ID

**Returns**: Generation status and result if complete

#### `extend_video_sequence`
**Description**: Extend existing video clips
**Parameters**:
- `video_data` (string, required): Base64 encoded video
- `extension_prompt` (string, required): Extension description
- `duration` (integer, optional): Extension length

**Returns**: Extended video generation job ID

## Agent Coordination Patterns

### Sequential Workflow
```
1. Music Research Agent:
   - search_music("Artist Name")
   - get_artist_profile(artist_name, include_all=True)
   - analyze_song_context(song_title, artist_name)

2. Visual Content Agent:
   - search_reference_images("artist era style")
   - generate_concept_image(concept, artist, song)
   - create_artist_visual(artist, "portrait")

3. Video Production Agent:
   - generate_music_video_clip(concept, artist, song)
   - check_video_status(job_id)
```

### Parallel Workflow
```
Concurrent execution:
- Music research (metadata + lyrics)
- Visual research (reference images)
- Concept development (initial ideas)

Then sequential:
- Concept refinement
- Image generation
- Video production
```

### Cross-Server Communication
```
# Music Research Server provides context
artist_data = await music_server.get_artist_profile(artist)

# Visual Content Server uses context
concept_image = await visual_server.generate_concept_image(
    concept=concept,
    artist_context=artist_data
)

# Video Production Server combines inputs
video = await video_server.generate_music_video_clip(
    concept=concept,
    visual_reference=concept_image,
    artist_data=artist_data
)
```

## Implementation Priorities

### Phase 1: Core MCP Servers
1. **Music Research Server** - Foundation for all workflows
2. **Visual Content Server** - Core creative capabilities
3. **Video Production Server** - Final output generation

### Phase 2: Enhanced Features
1. **Cross-server data sharing** - Shared context and state
2. **Workflow orchestration** - Automated agent coordination
3. **Result caching** - Performance optimization

### Phase 3: Advanced Capabilities
1. **Multi-agent collaboration** - Complex workflow support
2. **Quality assessment** - Automated output evaluation
3. **Style consistency** - Maintaining visual coherence

## Configuration Management

### Environment Variables
```bash
# Shared across all MCP servers
GOOGLE_API_KEY=...
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
LASTFM_API_KEY=...
GENIUS_ACCESS_TOKEN=...
GOOGLE_SEARCH_ENGINE_ID=...

# Server-specific settings
MUSIC_SERVER_PORT=8001
VISUAL_SERVER_PORT=8002
VIDEO_SERVER_PORT=8003

# Performance tuning
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=300
CACHE_TTL=3600
```

### Server Configuration
```python
# Each server has its own config
class ServerConfig:
    def __init__(self):
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_REQUESTS", 5))
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", 300))
        self.cache_ttl = int(os.getenv("CACHE_TTL", 3600))
        # API credentials loaded per client
```

## Testing Strategy

### Unit Tests
- Mock client responses
- Test tool parameter validation
- Error handling verification

### Integration Tests
- Real API calls with test data
- Cross-server communication
- End-to-end workflow testing

### Performance Tests
- Load testing with concurrent requests
- Memory usage monitoring
- Response time benchmarking

---

*This MCP server strategy provides a structured approach to exposing the Empire.AI API clients as tools for AI agents, enabling sophisticated music video creation workflows while maintaining clean separation of concerns and optimal performance.*
