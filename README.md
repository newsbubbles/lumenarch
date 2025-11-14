# Empire.AI Music Video Maker

A comprehensive system for creating high-end conceptual music videos using AI-powered research, image generation, and video creation.

## 🎯 Overview

Empire.AI combines multiple specialized API clients to research artists, analyze songs, generate visual concepts, and create professional music videos. The system uses a modular architecture with individual clients for each API service.

## 🏗️ Architecture

### API Clients

Each API service has its own dedicated client in the `clients/` directory:

- **`clients/spotify/`** - Spotify Web API for music metadata and audio analysis
- **`clients/lastfm/`** - Last.fm API for artist biographies and music tags
- **`clients/genius/`** - Genius API for lyrics and cultural context
- **`clients/gemini/`** - Gemini (NanoBanana) API for image generation and editing
- **`clients/google_search/`** - Google Custom Search API for reference images
- **`clients/veo/`** - Veo 3.1 API for high-quality video generation

### Features by Client

#### Music Research
- **Spotify**: Artist metadata, audio features (danceability, energy, valence), related artists
- **Last.fm**: Artist biographies, genre tags, similar artists, play statistics
- **Genius**: Complete lyrics, annotations, song meanings, cultural context

#### Visual Content
- **Gemini**: Text-to-image generation, image editing, concept art creation
- **Google Search**: Era-specific artist photos, reference images, visual research
- **Veo**: High-fidelity 8-second videos with audio, multiple generation modes

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and setup
git clone <repository-url>
cd empire-ai-music-video-maker

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. API Credentials

Edit `.env` with your API credentials:

- **Spotify**: Register at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
- **Last.fm**: Get API key at [Last.fm API](https://www.last.fm/api/account/create)
- **Genius**: Create app at [Genius API Clients](http://genius.com/api-clients)
- **Google**: Get API key at [Google AI Studio](https://aistudio.google.com/apikey)
- **Google Search**: Set up at [Programmable Search Engine](https://programmablesearchengine.google.com/controlpanel/all)

### 3. Basic Usage

```python
import asyncio
from clients.spotify import SpotifyClient, SearchRequest
from clients.gemini import GeminiImageClient, TextToImageRequest

async def create_concept_art():
    # Search for artist
    async with SpotifyClient() as spotify:
        search = SearchRequest(query="Radiohead", search_type=["artist"])
        results = await spotify.search(search)
        artist = results.artists[0] if results.artists else None
    
    # Generate concept image
    async with GeminiImageClient() as gemini:
        request = TextToImageRequest(
            prompt="Cinematic concept art for Radiohead, moody lighting, abstract",
            aspect_ratio="16:9",
            style="cinematic"
        )
        response = await gemini.generate_text_to_image(request)
        
        # Save the image
        if response.images:
            gemini.save_image(response.images[0], "radiohead_concept.png")

# Run the example
asyncio.run(create_concept_art())
```

## 📚 API Client Documentation

### Spotify Client

```python
from clients.spotify import SpotifyClient, SearchRequest, ArtistRequest

async with SpotifyClient() as client:
    # Search for artists
    search = SearchRequest(query="Artist Name", search_type=["artist"])
    results = await client.search(search)
    
    # Get artist details
    artist_req = ArtistRequest(artist_id="artist_id")
    artist = await client.get_artist(artist_req)
    
    # Get audio features
    features = await client.get_audio_features(
        AudioFeaturesRequest(track_id="track_id")
    )
```

### Gemini Image Client

```python
from clients.gemini import GeminiImageClient, TextToImageRequest

async with GeminiImageClient() as client:
    # Generate image from text
    request = TextToImageRequest(
        prompt="Your image description",
        aspect_ratio="16:9",
        style="photorealistic"
    )
    response = await client.generate_text_to_image(request)
    
    # Save the result
    client.save_image(response.images[0], "output.png")
```

### Veo Video Client

```python
from clients.veo import VeoVideoClient, TextToVideoRequest

async with VeoVideoClient() as client:
    # Generate video from text
    request = TextToVideoRequest(
        prompt="Your video description",
        config=VideoGenerationConfig(
            duration=8,
            aspect_ratio="16:9",
            resolution="1080p"
        )
    )
    
    # Start generation
    response = await client.generate_text_to_video(request)
    
    # Wait for completion
    result = await client.wait_for_completion(response.operation_name)
    
    # Save the video
    client.save_video(result.video, "output.mp4")
```

## 🔧 Configuration

### Rate Limiting

All clients implement intelligent rate limiting and retry logic:

- Exponential backoff on rate limit hits
- Automatic token refresh for OAuth2 clients
- Request queuing for high-volume usage

### Error Handling

- Comprehensive error messages for API failures
- Graceful degradation when services are unavailable
- Detailed logging for debugging

### Cost Management

- Built-in usage tracking for paid APIs
- Configurable spending limits
- Efficient caching to minimize API calls

## 📁 Project Structure

```
empire-ai-music-video-maker/
├── clients/                 # API client modules
│   ├── spotify/            # Spotify Web API client
│   ├── lastfm/             # Last.fm API client
│   ├── genius/             # Genius API client
│   ├── gemini/             # Gemini image generation client
│   ├── google_search/      # Google Custom Search client
│   └── veo/                # Veo video generation client
├── notes/                  # Research and development notes
│   ├── research/           # API research documentation
│   └── conversations/      # Development conversations
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## 🔮 Next Steps

This project provides the foundation API clients. Next phases include:

1. **MCP Server Development** - Wrap clients in Model Context Protocol servers
2. **Agent Architecture** - Create specialized agents for different workflows
3. **Data Pipeline** - Implement caching and data management
4. **User Interface** - Build web interface for project management

## 📖 Research Notes

Detailed API research and implementation notes are available in:

- [`notes/research/api_analysis.md`](notes/research/api_analysis.md) - Comprehensive API analysis
- [`notes/conversations/`](notes/conversations/) - Development session logs

## 🤝 Contributing

Each client is independently testable and follows consistent patterns:

- Async-first design with httpx
- Pydantic models for type safety
- Comprehensive error handling
- Environment-based configuration

## 📄 License

[Add your license here]

---

*Empire.AI - Transforming music into visual experiences through AI*
