# API Client Architecture - Empire.AI

*Created: 2025-11-14*

## Overview

The Empire.AI project implements a modular API client architecture where each external service has its own dedicated client module. This design provides clean separation of concerns, independent testing capabilities, and flexible integration options.

## Design Principles

### 1. Async-First Architecture
- All clients use `httpx.AsyncClient` for non-blocking operations
- Support for context managers (`async with`) for proper resource cleanup
- Async methods throughout for scalable concurrent operations

### 2. Type Safety with Pydantic
- Request and response models defined as Pydantic `BaseModel` classes
- Explicit field descriptions and validation
- Type hints throughout for IDE support and runtime validation

### 3. Consistent Error Handling
- Standardized exception patterns across all clients
- Graceful handling of API-specific errors
- Detailed error messages with context

### 4. Environment-Based Configuration
- API keys and secrets loaded from environment variables
- Fallback parameter support for flexibility
- Clear error messages for missing credentials

### 5. Modular Structure
- Each API service in its own subfolder under `clients/`
- Independent `requirements.txt` for each client
- Self-contained `__init__.py` exports

## Client Implementations

### Music Research Clients

#### Spotify Client (`clients/spotify/`)
**Authentication**: OAuth2 Client Credentials
**Key Features**:
- Automatic token management and refresh
- Comprehensive artist, album, and track search
- Audio feature analysis (danceability, energy, valence)
- Related artists discovery

**Core Models**:
- `SpotifyArtistFull` - Complete artist information
- `SpotifyAudioFeatures` - Musical analysis data
- `SearchRequest/Response` - Flexible search interface

**Usage Pattern**:
```python
async with SpotifyClient() as client:
    search = SearchRequest(query="artist", search_type=["artist"])
    results = await client.search(search)
```

#### Last.fm Client (`clients/lastfm/`)
**Authentication**: Simple API key
**Key Features**:
- Artist biographies and background information
- Music genre tags and classifications
- Similar artist recommendations
- Play statistics and listener counts

**Core Models**:
- `ArtistInfoResponse` - Complete artist profile
- `LastFmBiography` - Biographical information
- `LastFmTag` - Genre and style tags

**Usage Pattern**:
```python
async with LastFmClient() as client:
    request = ArtistInfoRequest(artist="Artist Name")
    info = await client.get_artist_info(request)
```

#### Genius Client (`clients/genius/`)
**Authentication**: OAuth2 Access Token
**Key Features**:
- Complete song lyrics retrieval
- Community annotations and interpretations
- Artist discography information
- Cultural context and song meanings

**Core Models**:
- `GeniusSong` - Complete song information with lyrics
- `GeniusAnnotation` - Community interpretations
- `SearchResponse` - Comprehensive search results

**Usage Pattern**:
```python
async with GeniusClient() as client:
    search = SearchRequest(query="song title artist")
    results = await client.search(search)
```

### Visual Content Clients

#### Gemini Image Client (`clients/gemini/`)
**Authentication**: Google API key
**Key Features**:
- Text-to-image generation with multiple aspect ratios
- Image editing and modification
- Multi-image composition
- Style-specific generation

**Core Models**:
- `TextToImageRequest` - Text-based generation
- `ImageEditRequest` - Image modification
- `GeneratedImage` - Base64 encoded results

**Usage Pattern**:
```python
async with GeminiImageClient() as client:
    request = TextToImageRequest(
        prompt="description",
        aspect_ratio="16:9"
    )
    response = await client.generate_text_to_image(request)
```

#### Google Custom Search Client (`clients/google_search/`)
**Authentication**: API key + Search Engine ID
**Key Features**:
- Image search with filters (size, type, color)
- Web search capabilities
- Era-specific artist photo discovery
- Reference image collection

**Core Models**:
- `ImageSearchRequest` - Image search parameters
- `SearchImage` - Image result with metadata
- `SearchResponse` - Comprehensive results

**Usage Pattern**:
```python
async with GoogleCustomSearchClient() as client:
    request = ImageSearchRequest(
        query="artist 1980s concert",
        img_type="photo"
    )
    results = await client.search_images(request)
```

#### Veo Video Client (`clients/veo/`)
**Authentication**: Google API key
**Key Features**:
- Text-to-video generation (8 seconds, 1080p)
- Image-to-video conversion
- Video extension capabilities
- Frame-directed generation

**Core Models**:
- `TextToVideoRequest` - Text-based video generation
- `VideoGenerationConfig` - Quality and format settings
- `GeneratedVideo` - Base64 encoded video results

**Usage Pattern**:
```python
async with VeoVideoClient() as client:
    request = TextToVideoRequest(
        prompt="video description",
        config=VideoGenerationConfig(duration=8, resolution="1080p")
    )
    response = await client.generate_text_to_video(request)
    result = await client.wait_for_completion(response.operation_name)
```

## Common Patterns

### Request/Response Models
All clients follow consistent patterns:
- `*Request` models for input parameters
- `*Response` models for structured output
- Optional configuration models for advanced settings

### Error Handling
```python
try:
    result = await client.some_operation(request)
except Exception as e:
    # All clients provide detailed error context
    logger.error(f"Operation failed: {e}")
```

### Resource Management
```python
# Automatic cleanup with context managers
async with ClientClass() as client:
    # Operations here
    pass  # Client automatically closed

# Or manual management
client = ClientClass()
try:
    # Operations
    pass
finally:
    await client.close()
```

## Performance Considerations

### Rate Limiting
- All clients implement intelligent retry logic
- Exponential backoff on rate limit hits
- Configurable timeout settings

### Caching Strategy
- Response caching at application level (not in clients)
- Token caching for OAuth2 clients
- Efficient pagination handling

### Memory Management
- Streaming support for large responses
- Base64 encoding for binary data (images/videos)
- Proper async resource cleanup

## Testing Strategy

### Unit Tests
- Mock HTTP responses for reliable testing
- Pydantic model validation tests
- Error handling verification

### Integration Tests
- Real API calls with test credentials
- Rate limiting behavior verification
- Authentication flow testing

### Example Test Structure
```python
@pytest.mark.asyncio
async def test_spotify_search():
    async with SpotifyClient() as client:
        request = SearchRequest(query="test", search_type=["artist"])
        response = await client.search(request)
        assert isinstance(response, SearchResponse)
```

## Security Considerations

### Credential Management
- Environment variables for all secrets
- No hardcoded API keys or tokens
- Clear documentation for credential setup

### Data Handling
- No persistent storage of user data
- Secure transmission (HTTPS only)
- Minimal data retention

### API Key Rotation
- Support for runtime credential updates
- Graceful handling of expired tokens
- Clear error messages for authentication issues

## Future Enhancements

### Monitoring and Observability
- Request/response logging
- Performance metrics collection
- Usage tracking and cost monitoring

### Advanced Features
- Request queuing and batching
- Circuit breaker patterns
- Fallback service support

### Optimization
- Connection pooling
- Response compression
- Intelligent caching strategies

## Integration with MCP Servers

The client architecture is designed to be easily wrapped in MCP (Model Context Protocol) servers:

### Grouping Strategy
1. **Music Research Server** - Spotify, Last.fm, Genius clients
2. **Visual Content Server** - Gemini, Google Search clients  
3. **Video Production Server** - Veo client

### MCP Tool Mapping
- Each client method becomes an MCP tool
- Request models become tool parameters
- Response models become tool outputs

### Agent Coordination
- Clients provide the foundation for agent workflows
- MCP servers enable cross-agent communication
- Shared data models ensure consistency

---

*This architecture provides a solid foundation for building sophisticated music video creation workflows while maintaining clean separation of concerns and maximum flexibility.*
