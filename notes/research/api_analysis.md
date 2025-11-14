# API Research Analysis - Empire.AI Music Video Maker

*Research Date: 2025-11-14*

## Project Overview

Empire.AI is a comprehensive music video creation system that uses multiple specialized agents to research artists, analyze songs, generate visual concepts, and create high-end music videos. The system requires integration with multiple APIs across music research, image generation, and video creation domains.

## Music Research APIs

### Spotify Web API
**Documentation**: https://developer.spotify.com/documentation/web-api/

**Capabilities**:
- Artist metadata (biography, genres, popularity)
- Album and track information
- Audio analysis features (danceability, energy, valence, tempo, key, mode)
- Related artists discovery
- Search across entire Spotify catalog
- Artist top tracks and albums

**Authentication**: OAuth2 Client Credentials flow
- Requires app registration at https://developer.spotify.com/dashboard
- Client ID and Client Secret needed
- Access token expires in 1 hour

**Rate Limits**:
- Standard rate limiting applies
- Retry-After header provided on rate limit hits

**Key Endpoints**:
- `/v1/search` - Search for artists, albums, tracks
- `/v1/artists/{id}` - Get artist details
- `/v1/artists/{id}/albums` - Get artist albums
- `/v1/artists/{id}/top-tracks` - Get artist top tracks
- `/v1/artists/{id}/related-artists` - Get related artists
- `/v1/audio-features/{id}` - Get audio analysis

**Best For**: Comprehensive music metadata, audio analysis for mood matching

### Last.fm API
**Documentation**: https://www.last.fm/api/show/artist.getInfo

**Capabilities**:
- Artist biographies and background information
- Music tags and genre classification
- Similar artists recommendations
- Artist statistics (listeners, play count)
- Artist images in multiple sizes

**Authentication**: API key only (no OAuth required)
- Register at https://www.last.fm/api/account/create
- Simple API key authentication

**Rate Limits**:
- Standard rate limiting (error code 29 when exceeded)
- No specific limits documented

**Key Endpoints**:
- `artist.getInfo` - Get artist metadata and biography
- `artist.getSimilar` - Get similar artists
- `artist.getTopTracks` - Get artist top tracks
- `artist.getTags` - Get artist tags

**Limitations**:
- Biography text truncated at 300 characters in summary
- Full biography available in separate field

**Best For**: Artist background research, genre analysis, biographical context

### Genius API
**Documentation**: https://docs.genius.com/

**Capabilities**:
- Complete song lyrics
- Community annotations explaining lyric meanings
- Artist information and discographies
- Song and artist search
- Annotation management (create, update, delete)

**Authentication**: OAuth2
- Register application at http://genius.com/api-clients
- Supports both user access tokens and client access tokens
- Client access tokens for read-only access

**Rate Limits**:
- Standard rate limiting applies
- No specific limits documented

**Key Endpoints**:
- `/search` - Search Genius content
- `/songs/{id}` - Get song data including lyrics
- `/artists/{id}` - Get artist data
- `/artists/{id}/songs` - Get songs by artist
- `/annotations/{id}` - Get annotation data

**Best For**: Lyric analysis, song meaning research, cultural context

## Image Generation & Search APIs

### Gemini (NanoBanana) Image API
**Documentation**: https://ai.google.dev/gemini-api/docs/image-generation

**Capabilities**:
- Text-to-image generation
- Image editing and modification
- Multi-image to image composition
- Iterative refinement through conversation
- High-fidelity text rendering in images

**Authentication**: Google API key
- Obtain from https://aistudio.google.com/apikey
- Use "x-goog-api-key" header

**Endpoints**:
- `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent`

**Supported Formats**:
- Input: PNG, JPEG (base64 encoded)
- Output: PNG format
- Multiple aspect ratios (1:1, 16:9, 9:16, etc.)

**Pricing**: ~$30 per 1M tokens (~$0.04 per image)

**Limitations**:
- Higher latency than some alternatives
- Generated images may be watermarked
- Best performance with English prompts

**Best For**: Conceptual image generation, artistic photo editing

### Google Custom Search API
**Documentation**: https://developers.google.com/custom-search/v1/overview

**Capabilities**:
- Image search results from Google
- Web search results
- Configurable search scopes
- JSON formatted results

**Authentication**: 
- API key from Google Cloud Console
- Search Engine ID from Programmable Search Engine setup

**Pricing**:
- 100 free queries per day
- $5 per 1000 queries after free tier
- Maximum 10,000 queries per day

**Setup Requirements**:
1. Create Programmable Search Engine
2. Obtain Search Engine ID
3. Get API key from Cloud Console

**Best For**: Finding era-specific artist photos, reference images

## Video Generation APIs

### Veo 3.1 Video API
**Documentation**: https://ai.google.dev/gemini-api/docs/video

**Capabilities**:
- High-fidelity 8-second videos (720p/1080p)
- Native audio generation
- Text-to-video generation
- Image-to-video generation
- Video extension (up to 20 extensions of 7 seconds each)
- Frame-specific generation

**Authentication**: Google API key
- Same key as Gemini API

**Endpoints**:
- `/models/veo-3.1-generate-preview:predictLongRunning`
- Operations endpoint for status checking

**Output Formats**:
- Resolution: 720p (default) or 1080p
- Aspect ratio: 16:9 (default) or 9:16
- Frame rate: 24fps
- Duration: 4, 6, or 8 seconds
- Format: MP4 with synchronized audio

**Limitations**:
- Generation latency: 11 seconds minimum, up to 6 minutes
- Video retention: Only 2 days on server
- Regional restrictions in EU/UK/CH/MENA
- Generated videos may be watermarked

**Best For**: Music video sequences, cinematic shots, promotional content

## Implementation Strategy

### Authentication Management
1. **Environment Variables**: Store all API keys and secrets securely
2. **Token Management**: Implement refresh logic for OAuth2 tokens
3. **Fallback Strategies**: Handle authentication failures gracefully

### Rate Limiting Strategy
1. **Intelligent Caching**: Cache API responses to reduce requests
2. **Background Queues**: Use async processing for slow operations
3. **Batch Processing**: Group requests where possible
4. **Retry Logic**: Implement exponential backoff

### Cost Management
1. **Usage Monitoring**: Track API costs in real-time
2. **Budget Alerts**: Set spending limits
3. **Efficient Prompting**: Optimize prompts for better results with fewer attempts
4. **Result Caching**: Store generated content to avoid regeneration

### Error Handling
1. **Graceful Degradation**: Fallback to alternative APIs when primary fails
2. **User Feedback**: Clear error messages for different failure modes
3. **Logging**: Comprehensive logging for debugging
4. **Monitoring**: Health checks for all external services

## Next Steps

1. **API Client Development**: Create individual clients for each API
2. **Data Schema Design**: Define data models for storing API responses
3. **Agent Architecture**: Design how agents will coordinate API usage
4. **MCP Server Structure**: Plan how to expose functionality through MCP
5. **Testing Strategy**: Create comprehensive test suites for all integrations

## Risk Assessment

### High Risk
- **Veo 3.1**: High latency, limited retention, potential high costs
- **Gemini**: Cost per image could scale quickly with heavy usage

### Medium Risk
- **Google Custom Search**: Limited free tier, daily caps
- **Spotify**: OAuth complexity, rate limiting

### Low Risk
- **Last.fm**: Simple authentication, stable API
- **Genius**: Well-documented, predictable costs

## Recommended Development Order

1. **Last.fm Client** (simplest authentication)
2. **Spotify Client** (OAuth2 implementation)
3. **Genius Client** (extend OAuth2 patterns)
4. **Google Custom Search Client** (API key + additional setup)
5. **Gemini Client** (image generation)
6. **Veo Client** (video generation, most complex)
