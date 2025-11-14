"""Music Research MCP Server for Empire.AI

Provides comprehensive music research capabilities through Spotify, Last.fm, and Genius APIs.
Exposes tools for artist discovery, audio analysis, and lyrical content research.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Any, Optional, List, Dict, Literal
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context

# Import our API clients
from clients.spotify.client import (
    SpotifyClient, SearchRequest, SearchResponse, ArtistRequest, 
    ArtistAlbumsRequest, ArtistTopTracksRequest, RelatedArtistsRequest,
    AudioFeaturesRequest, SpotifyArtistFull, SpotifyAlbumSimple, 
    SpotifyTrackFull, SpotifyAudioFeatures
)
from clients.lastfm.client import (
    LastFmClient, ArtistInfoRequest, ArtistInfoResponse,
    SimilarArtistsRequest, SimilarArtistsResponse, TopTagsRequest,
    TopTagsResponse
)
from clients.genius.client import (
    GeniusClient, SearchRequest as GeniusSearchRequest, 
    SearchResponse as GeniusSearchResponse, SongRequest,
    GeniusSong
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
logger.info("Music Research MCP server starting")

# Request/Response Models for MCP Tools

class MusicSearchRequest(BaseModel):
    """Request for unified music search across platforms."""
    query: str = Field(description="Search query for music content")
    platforms: List[Literal["spotify", "lastfm", "genius"]] = Field(
        ["spotify"], description="Platforms to search on"
    )
    content_types: List[Literal["artist", "album", "track"]] = Field(
        ["artist"], description="Content types to search for"
    )
    limit: int = Field(10, description="Maximum results per platform")
    market: Optional[str] = Field(None, description="Market/country code for Spotify")

class MusicSearchResponse(BaseModel):
    """Unified search results from multiple platforms."""
    query: str = Field(description="Original search query")
    spotify_results: Optional[SearchResponse] = Field(None, description="Spotify search results")
    lastfm_results: Optional[Dict[str, Any]] = Field(None, description="Last.fm search results")
    genius_results: Optional[GeniusSearchResponse] = Field(None, description="Genius search results")
    total_results: int = Field(description="Total number of results found")

class ArtistProfileRequest(BaseModel):
    """Request for comprehensive artist profile."""
    artist_name: str = Field(description="Artist name to research")
    include_audio_analysis: bool = Field(True, description="Include Spotify audio features analysis")
    include_biography: bool = Field(True, description="Include Last.fm biography and tags")
    include_similar_artists: bool = Field(True, description="Include related/similar artists")
    include_top_tracks: bool = Field(True, description="Include artist's top tracks")
    market: str = Field("US", description="Market for Spotify data")

class ArtistProfileResponse(BaseModel):
    """Comprehensive artist profile from multiple sources."""
    artist_name: str = Field(description="Artist name")
    spotify_data: Optional[SpotifyArtistFull] = Field(None, description="Spotify artist information")
    lastfm_data: Optional[ArtistInfoResponse] = Field(None, description="Last.fm artist information")
    top_tracks: Optional[List[SpotifyTrackFull]] = Field(None, description="Top tracks from Spotify")
    related_artists: Optional[List[SpotifyArtistFull]] = Field(None, description="Related artists")
    similar_artists: Optional[SimilarArtistsResponse] = Field(None, description="Similar artists from Last.fm")
    audio_features_summary: Optional[Dict[str, float]] = Field(None, description="Average audio features")
    genres_and_tags: Optional[List[str]] = Field(None, description="Combined genres and tags")

class SongAnalysisRequest(BaseModel):
    """Request for deep song analysis."""
    song_title: str = Field(description="Song title")
    artist_name: str = Field(description="Artist name")
    include_lyrics: bool = Field(True, description="Include lyrics from Genius")
    include_annotations: bool = Field(True, description="Include Genius annotations")
    include_audio_features: bool = Field(True, description="Include Spotify audio features")
    market: str = Field("US", description="Market for Spotify data")

class SongAnalysisResponse(BaseModel):
    """Comprehensive song analysis."""
    song_title: str = Field(description="Song title")
    artist_name: str = Field(description="Artist name")
    spotify_track: Optional[SpotifyTrackFull] = Field(None, description="Spotify track information")
    audio_features: Optional[SpotifyAudioFeatures] = Field(None, description="Spotify audio features")
    genius_song: Optional[GeniusSong] = Field(None, description="Genius song data with lyrics")
    lyrics_summary: Optional[str] = Field(None, description="Summary of lyrical themes")
    musical_characteristics: Optional[Dict[str, Any]] = Field(None, description="Musical analysis")

class AudioFeaturesAnalysisRequest(BaseModel):
    """Request for audio features analysis."""
    track_ids: List[str] = Field(description="List of Spotify track IDs to analyze")
    include_summary: bool = Field(True, description="Include statistical summary")

class AudioFeaturesAnalysisResponse(BaseModel):
    """Audio features analysis results."""
    track_features: List[SpotifyAudioFeatures] = Field(description="Individual track features")
    summary_stats: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Statistical summary (mean, min, max, std)"
    )
    insights: Optional[List[str]] = Field(None, description="Musical insights and patterns")

class SimilarArtistsDiscoveryRequest(BaseModel):
    """Request for discovering similar artists."""
    artist_name: str = Field(description="Base artist name")
    discovery_depth: Literal["shallow", "deep"] = Field(
        "shallow", description="Depth of discovery (shallow=direct, deep=multi-hop)"
    )
    max_results: int = Field(20, description="Maximum number of similar artists to return")
    include_audio_similarity: bool = Field(
        True, description="Include audio feature similarity analysis"
    )

class SimilarArtistsDiscoveryResponse(BaseModel):
    """Similar artists discovery results."""
    base_artist: str = Field(description="Original artist name")
    spotify_related: Optional[List[SpotifyArtistFull]] = Field(
        None, description="Spotify related artists"
    )
    lastfm_similar: Optional[List[Dict[str, Any]]] = Field(
        None, description="Last.fm similar artists"
    )
    combined_recommendations: Optional[List[Dict[str, Any]]] = Field(
        None, description="Combined and scored recommendations"
    )
    discovery_insights: Optional[List[str]] = Field(
        None, description="Insights about musical connections"
    )

# MCP Context for persistent client connections
class MusicResearchContext:
    """Context for music research clients."""
    def __init__(self, spotify_client: SpotifyClient, lastfm_client: LastFmClient, genius_client: GeniusClient):
        self.spotify_client = spotify_client
        self.lastfm_client = lastfm_client
        self.genius_client = genius_client

# Lifespan context manager for API clients
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[Any]:
    """Initialize and manage API client connections."""
    try:
        # Initialize clients
        spotify_client = SpotifyClient()
        lastfm_client = LastFmClient()
        genius_client = GeniusClient()
        
        # Enter async context managers
        await spotify_client.__aenter__()
        await lastfm_client.__aenter__()
        await genius_client.__aenter__()
        
        logger.info("Music research clients initialized successfully")
        
        yield MusicResearchContext(
            spotify_client=spotify_client,
            lastfm_client=lastfm_client,
            genius_client=genius_client
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize music research clients: {str(e)}")
        raise ValueError(f"Failed to initialize music research clients: {str(e)}")
    finally:
        # Clean up client connections
        try:
            await spotify_client.__aexit__(None, None, None)
            await lastfm_client.__aexit__(None, None, None)
            await genius_client.__aexit__(None, None, None)
            logger.info("Music research clients closed successfully")
        except Exception as e:
            logger.error(f"Error closing clients: {str(e)}")

# Initialize FastMCP server
mcp = FastMCP("Music Research Server", lifespan=lifespan)

# MCP Tools

@mcp.tool()
async def music_search(request: MusicSearchRequest, ctx: Context) -> MusicSearchResponse:
    """Search for music content across multiple platforms (Spotify, Last.fm, Genius).
    
    Performs unified search across selected platforms and returns comprehensive results.
    Useful for initial research and discovery of artists, albums, and tracks.
    """
    try:
        await ctx.info(f"Searching for '{request.query}' on platforms: {request.platforms}")
        
        clients = ctx.request_context.lifespan_context
        response = MusicSearchResponse(
            query=request.query,
            total_results=0
        )
        
        # Search Spotify
        if "spotify" in request.platforms:
            try:
                spotify_search = SearchRequest(
                    query=request.query,
                    search_type=request.content_types,
                    limit=request.limit,
                    market=request.market
                )
                spotify_results = await clients.spotify_client.search(spotify_search)
                response.spotify_results = spotify_results
                
                # Count results
                count = 0
                if spotify_results.artists:
                    count += len(spotify_results.artists)
                if spotify_results.albums:
                    count += len(spotify_results.albums)
                if spotify_results.tracks:
                    count += len(spotify_results.tracks)
                response.total_results += count
                
                logger.info(f"Found {count} Spotify results for '{request.query}'")
                
            except Exception as e:
                logger.error(f"Spotify search failed: {str(e)}")
                await ctx.info(f"Spotify search failed: {str(e)}")
        
        # Search Last.fm (artist info if searching for artists)
        if "lastfm" in request.platforms and "artist" in request.content_types:
            try:
                lastfm_search = ArtistInfoRequest(artist=request.query)
                lastfm_results = await clients.lastfm_client.get_artist_info(lastfm_search)
                response.lastfm_results = lastfm_results.model_dump()
                response.total_results += 1
                
                logger.info(f"Found Last.fm artist info for '{request.query}'")
                
            except Exception as e:
                logger.error(f"Last.fm search failed: {str(e)}")
                await ctx.info(f"Last.fm search failed: {str(e)}")
        
        # Search Genius
        if "genius" in request.platforms:
            try:
                genius_search = GeniusSearchRequest(query=request.query)
                genius_results = await clients.genius_client.search(genius_search)
                response.genius_results = genius_results
                response.total_results += len(genius_results.hits)
                
                logger.info(f"Found {len(genius_results.hits)} Genius results for '{request.query}'")
                
            except Exception as e:
                logger.error(f"Genius search failed: {str(e)}")
                await ctx.info(f"Genius search failed: {str(e)}")
        
        await ctx.info(f"Search completed. Total results: {response.total_results}")
        return response
        
    except Exception as e:
        logger.error(f"Music search failed: {str(e)}")
        raise ValueError(f"Music search failed: {str(e)}")

@mcp.tool()
async def music_get_artist_profile(request: ArtistProfileRequest, ctx: Context) -> ArtistProfileResponse:
    """Get comprehensive artist profile combining data from Spotify, Last.fm, and related sources.
    
    Provides detailed artist information including biography, genres, popularity metrics,
    top tracks, related artists, and audio characteristics analysis.
    """
    try:
        await ctx.info(f"Building comprehensive profile for artist: {request.artist_name}")
        
        clients = ctx.request_context.lifespan_context
        response = ArtistProfileResponse(artist_name=request.artist_name)
        
        # Search for artist on Spotify first
        spotify_search = SearchRequest(
            query=request.artist_name,
            search_type=["artist"],
            limit=1,
            market=request.market
        )
        spotify_search_results = await clients.spotify_client.search(spotify_search)
        
        if spotify_search_results.artists and len(spotify_search_results.artists) > 0:
            artist = spotify_search_results.artists[0]
            response.spotify_data = artist
            artist_id = artist.id
            
            await ctx.info(f"Found Spotify artist: {artist.name} (ID: {artist_id})")
            
            # Get top tracks if requested
            if request.include_top_tracks:
                try:
                    top_tracks_req = ArtistTopTracksRequest(
                        artist_id=artist_id,
                        market=request.market
                    )
                    top_tracks = await clients.spotify_client.get_artist_top_tracks(top_tracks_req)
                    response.top_tracks = top_tracks
                    
                    await ctx.info(f"Retrieved {len(top_tracks)} top tracks")
                    
                except Exception as e:
                    logger.error(f"Failed to get top tracks: {str(e)}")
            
            # Get related artists if requested
            if request.include_similar_artists:
                try:
                    related_req = RelatedArtistsRequest(artist_id=artist_id)
                    related_artists = await clients.spotify_client.get_related_artists(related_req)
                    response.related_artists = related_artists
                    
                    await ctx.info(f"Retrieved {len(related_artists)} related artists")
                    
                except Exception as e:
                    logger.error(f"Failed to get related artists: {str(e)}")
            
            # Analyze audio features if requested
            if request.include_audio_analysis and response.top_tracks:
                try:
                    audio_features = []
                    for track in response.top_tracks[:5]:  # Analyze top 5 tracks
                        try:
                            features_req = AudioFeaturesRequest(track_id=track.id)
                            features = await clients.spotify_client.get_audio_features(features_req)
                            audio_features.append(features)
                        except Exception:
                            continue
                    
                    if audio_features:
                        # Calculate average features
                        avg_features = {
                            "danceability": sum(f.danceability for f in audio_features) / len(audio_features),
                            "energy": sum(f.energy for f in audio_features) / len(audio_features),
                            "valence": sum(f.valence for f in audio_features) / len(audio_features),
                            "acousticness": sum(f.acousticness for f in audio_features) / len(audio_features),
                            "instrumentalness": sum(f.instrumentalness for f in audio_features) / len(audio_features),
                            "tempo": sum(f.tempo for f in audio_features) / len(audio_features)
                        }
                        response.audio_features_summary = avg_features
                        
                        await ctx.info(f"Analyzed audio features for {len(audio_features)} tracks")
                        
                except Exception as e:
                    logger.error(f"Failed to analyze audio features: {str(e)}")
        
        # Get Last.fm data if requested
        if request.include_biography:
            try:
                lastfm_req = ArtistInfoRequest(artist=request.artist_name)
                lastfm_data = await clients.lastfm_client.get_artist_info(lastfm_req)
                response.lastfm_data = lastfm_data
                
                await ctx.info("Retrieved Last.fm biography and tags")
                
                # Get similar artists from Last.fm
                if request.include_similar_artists:
                    try:
                        similar_req = SimilarArtistsRequest(artist=request.artist_name, limit=10)
                        similar_artists = await clients.lastfm_client.get_similar_artists(similar_req)
                        response.similar_artists = similar_artists
                        
                        await ctx.info(f"Retrieved {len(similar_artists.similar_artists)} similar artists from Last.fm")
                        
                    except Exception as e:
                        logger.error(f"Failed to get similar artists from Last.fm: {str(e)}")
                
            except Exception as e:
                logger.error(f"Failed to get Last.fm data: {str(e)}")
        
        # Combine genres and tags
        genres_and_tags = []
        if response.spotify_data and response.spotify_data.genres:
            genres_and_tags.extend(response.spotify_data.genres)
        if response.lastfm_data and response.lastfm_data.top_tags:
            genres_and_tags.extend([tag.name for tag in response.lastfm_data.top_tags[:10]])
        
        if genres_and_tags:
            response.genres_and_tags = list(set(genres_and_tags))  # Remove duplicates
        
        await ctx.info("Artist profile compilation completed")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get artist profile: {str(e)}")
        raise ValueError(f"Failed to get artist profile: {str(e)}")

@mcp.tool()
async def music_analyze_song(request: SongAnalysisRequest, ctx: Context) -> SongAnalysisResponse:
    """Perform deep analysis of a specific song including lyrics, audio features, and cultural context.
    
    Combines Spotify's audio analysis with Genius's lyrical content and annotations
    to provide comprehensive song insights.
    """
    try:
        await ctx.info(f"Analyzing song: '{request.song_title}' by {request.artist_name}")
        
        clients = ctx.request_context.lifespan_context
        response = SongAnalysisResponse(
            song_title=request.song_title,
            artist_name=request.artist_name
        )
        
        # Search for track on Spotify
        spotify_search = SearchRequest(
            query=f"{request.song_title} {request.artist_name}",
            search_type=["track"],
            limit=1,
            market=request.market
        )
        spotify_results = await clients.spotify_client.search(spotify_search)
        
        if spotify_results.tracks and len(spotify_results.tracks) > 0:
            track = spotify_results.tracks[0]
            response.spotify_track = track
            
            await ctx.info(f"Found Spotify track: {track.name} by {track.artists[0].name}")
            
            # Get audio features if requested
            if request.include_audio_features:
                try:
                    features_req = AudioFeaturesRequest(track_id=track.id)
                    audio_features = await clients.spotify_client.get_audio_features(features_req)
                    response.audio_features = audio_features
                    
                    # Create musical characteristics summary
                    characteristics = {
                        "mood": "upbeat" if audio_features.valence > 0.6 else "melancholic" if audio_features.valence < 0.4 else "neutral",
                        "energy_level": "high" if audio_features.energy > 0.7 else "low" if audio_features.energy < 0.3 else "moderate",
                        "danceability": "very danceable" if audio_features.danceability > 0.7 else "not danceable" if audio_features.danceability < 0.3 else "moderately danceable",
                        "tempo_category": "fast" if audio_features.tempo > 120 else "slow" if audio_features.tempo < 80 else "moderate",
                        "acoustic_nature": "acoustic" if audio_features.acousticness > 0.5 else "electronic",
                        "key": audio_features.key,
                        "mode": "major" if audio_features.mode == 1 else "minor"
                    }
                    response.musical_characteristics = characteristics
                    
                    await ctx.info("Analyzed audio features and musical characteristics")
                    
                except Exception as e:
                    logger.error(f"Failed to get audio features: {str(e)}")
        
        # Get lyrics and annotations from Genius
        if request.include_lyrics:
            try:
                genius_search = GeniusSearchRequest(
                    query=f"{request.song_title} {request.artist_name}"
                )
                genius_results = await clients.genius_client.search(genius_search)
                
                if genius_results.hits and len(genius_results.hits) > 0:
                    # Get the first hit that looks like a match
                    for hit in genius_results.hits:
                        if (request.artist_name.lower() in hit.result.primary_artist.name.lower() or
                            hit.result.primary_artist.name.lower() in request.artist_name.lower()):
                            
                            song_req = SongRequest(song_id=hit.result.id)
                            genius_song = await clients.genius_client.get_song(song_req)
                            response.genius_song = genius_song
                            
                            await ctx.info(f"Retrieved lyrics and annotations from Genius")
                            
                            # Create lyrics summary if we have lyrics
                            if genius_song.lyrics:
                                # Simple analysis of lyrical themes
                                lyrics_lower = genius_song.lyrics.lower()
                                themes = []
                                
                                if any(word in lyrics_lower for word in ["love", "heart", "baby", "kiss"]):
                                    themes.append("love/romance")
                                if any(word in lyrics_lower for word in ["party", "dance", "night", "club"]):
                                    themes.append("party/celebration")
                                if any(word in lyrics_lower for word in ["pain", "hurt", "cry", "sad"]):
                                    themes.append("sadness/pain")
                                if any(word in lyrics_lower for word in ["money", "rich", "cash", "gold"]):
                                    themes.append("wealth/materialism")
                                if any(word in lyrics_lower for word in ["dream", "hope", "future", "believe"]):
                                    themes.append("hope/aspiration")
                                
                                if themes:
                                    response.lyrics_summary = f"Main themes: {', '.join(themes)}"
                                else:
                                    response.lyrics_summary = "Themes not clearly categorized from analysis"
                            
                            break
                
            except Exception as e:
                logger.error(f"Failed to get Genius data: {str(e)}")
        
        await ctx.info("Song analysis completed")
        return response
        
    except Exception as e:
        logger.error(f"Failed to analyze song: {str(e)}")
        raise ValueError(f"Failed to analyze song: {str(e)}")

@mcp.tool()
async def music_analyze_audio_features(request: AudioFeaturesAnalysisRequest, ctx: Context) -> AudioFeaturesAnalysisResponse:
    """Analyze audio features for multiple tracks to identify patterns and insights.
    
    Useful for understanding the musical characteristics of an artist's catalog
    or comparing tracks for playlist curation.
    """
    try:
        await ctx.info(f"Analyzing audio features for {len(request.track_ids)} tracks")
        
        clients = ctx.request_context.lifespan_context
        track_features = []
        
        # Get features for each track
        for track_id in request.track_ids:
            try:
                features_req = AudioFeaturesRequest(track_id=track_id)
                features = await clients.spotify_client.get_audio_features(features_req)
                track_features.append(features)
            except Exception as e:
                logger.error(f"Failed to get features for track {track_id}: {str(e)}")
                continue
        
        response = AudioFeaturesAnalysisResponse(track_features=track_features)
        
        if request.include_summary and track_features:
            # Calculate statistical summary
            import statistics
            
            features_to_analyze = [
                "acousticness", "danceability", "energy", "instrumentalness",
                "liveness", "loudness", "speechiness", "tempo", "valence"
            ]
            
            summary_stats = {}
            for feature in features_to_analyze:
                values = [getattr(f, feature) for f in track_features]
                summary_stats[feature] = {
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0
                }
            
            response.summary_stats = summary_stats
            
            # Generate insights
            insights = []
            
            # Energy analysis
            avg_energy = summary_stats["energy"]["mean"]
            if avg_energy > 0.7:
                insights.append("High-energy collection with driving, forceful tracks")
            elif avg_energy < 0.3:
                insights.append("Low-energy collection with calm, relaxed tracks")
            
            # Valence analysis
            avg_valence = summary_stats["valence"]["mean"]
            if avg_valence > 0.6:
                insights.append("Generally positive, upbeat emotional tone")
            elif avg_valence < 0.4:
                insights.append("Generally melancholic or sad emotional tone")
            
            # Danceability analysis
            avg_dance = summary_stats["danceability"]["mean"]
            if avg_dance > 0.7:
                insights.append("Highly danceable collection suitable for movement")
            elif avg_dance < 0.3:
                insights.append("Low danceability, more suitable for listening")
            
            # Acousticness analysis
            avg_acoustic = summary_stats["acousticness"]["mean"]
            if avg_acoustic > 0.5:
                insights.append("Predominantly acoustic instrumentation")
            else:
                insights.append("Predominantly electronic/produced instrumentation")
            
            # Tempo analysis
            avg_tempo = summary_stats["tempo"]["mean"]
            if avg_tempo > 140:
                insights.append("Fast-paced tracks with high BPM")
            elif avg_tempo < 80:
                insights.append("Slow-paced tracks with low BPM")
            
            response.insights = insights
            
            await ctx.info(f"Generated {len(insights)} musical insights")
        
        await ctx.info(f"Audio features analysis completed for {len(track_features)} tracks")
        return response
        
    except Exception as e:
        logger.error(f"Failed to analyze audio features: {str(e)}")
        raise ValueError(f"Failed to analyze audio features: {str(e)}")

@mcp.tool()
async def music_discover_similar_artists(request: SimilarArtistsDiscoveryRequest, ctx: Context) -> SimilarArtistsDiscoveryResponse:
    """Discover similar artists using multiple data sources and similarity algorithms.
    
    Combines Spotify's related artists with Last.fm's similar artists to provide
    comprehensive artist discovery with optional audio feature similarity analysis.
    """
    try:
        await ctx.info(f"Discovering artists similar to: {request.artist_name}")
        
        clients = ctx.request_context.lifespan_context
        response = SimilarArtistsDiscoveryResponse(base_artist=request.artist_name)
        
        # Get Spotify related artists
        try:
            # First find the artist
            spotify_search = SearchRequest(
                query=request.artist_name,
                search_type=["artist"],
                limit=1
            )
            search_results = await clients.spotify_client.search(spotify_search)
            
            if search_results.artists and len(search_results.artists) > 0:
                base_artist = search_results.artists[0]
                
                # Get related artists
                related_req = RelatedArtistsRequest(artist_id=base_artist.id)
                related_artists = await clients.spotify_client.get_related_artists(related_req)
                response.spotify_related = related_artists[:request.max_results]
                
                await ctx.info(f"Found {len(related_artists)} related artists from Spotify")
                
        except Exception as e:
            logger.error(f"Failed to get Spotify related artists: {str(e)}")
        
        # Get Last.fm similar artists
        try:
            similar_req = SimilarArtistsRequest(
                artist=request.artist_name,
                limit=min(request.max_results, 50)
            )
            similar_response = await clients.lastfm_client.get_similar_artists(similar_req)
            
            # Convert to dict format for response
            lastfm_similar = []
            for artist in similar_response.similar_artists:
                lastfm_similar.append({
                    "name": artist.name,
                    "match_score": artist.match,
                    "url": artist.url,
                    "playcount": artist.playcount,
                    "listeners": artist.listeners
                })
            
            response.lastfm_similar = lastfm_similar
            
            await ctx.info(f"Found {len(lastfm_similar)} similar artists from Last.fm")
            
        except Exception as e:
            logger.error(f"Failed to get Last.fm similar artists: {str(e)}")
        
        # Combine and score recommendations
        combined_recommendations = []
        seen_artists = set()
        
        # Add Spotify related artists
        if response.spotify_related:
            for artist in response.spotify_related:
                if artist.name.lower() not in seen_artists:
                    combined_recommendations.append({
                        "name": artist.name,
                        "source": "spotify",
                        "popularity": artist.popularity,
                        "genres": artist.genres,
                        "followers": artist.followers.total,
                        "confidence_score": 0.8  # Spotify's algorithm is generally reliable
                    })
                    seen_artists.add(artist.name.lower())
        
        # Add Last.fm similar artists
        if response.lastfm_similar:
            for artist_data in response.lastfm_similar:
                if artist_data["name"].lower() not in seen_artists:
                    combined_recommendations.append({
                        "name": artist_data["name"],
                        "source": "lastfm",
                        "match_score": artist_data["match_score"],
                        "playcount": artist_data["playcount"],
                        "listeners": artist_data["listeners"],
                        "confidence_score": artist_data["match_score"]  # Use Last.fm's match score
                    })
                    seen_artists.add(artist_data["name"].lower())
                else:
                    # Artist found in both sources - increase confidence
                    for rec in combined_recommendations:
                        if rec["name"].lower() == artist_data["name"].lower():
                            rec["confidence_score"] = min(0.95, rec["confidence_score"] + 0.15)
                            rec["sources"] = [rec.get("source", "spotify"), "lastfm"]
                            break
        
        # Sort by confidence score
        combined_recommendations.sort(key=lambda x: x["confidence_score"], reverse=True)
        response.combined_recommendations = combined_recommendations[:request.max_results]
        
        # Generate discovery insights
        insights = []
        
        if response.spotify_related and response.lastfm_similar:
            overlap = len([r for r in combined_recommendations if "sources" in r])
            insights.append(f"Found {overlap} artists recommended by both Spotify and Last.fm")
        
        if response.spotify_related:
            common_genres = {}
            for artist in response.spotify_related:
                for genre in artist.genres:
                    common_genres[genre] = common_genres.get(genre, 0) + 1
            
            if common_genres:
                top_genre = max(common_genres, key=common_genres.get)
                insights.append(f"Most common genre among similar artists: {top_genre}")
        
        total_found = len(combined_recommendations)
        insights.append(f"Total unique similar artists discovered: {total_found}")
        
        if request.discovery_depth == "deep" and total_found > 5:
            insights.append("Deep discovery mode: Consider exploring artists with lower confidence scores for hidden gems")
        
        response.discovery_insights = insights
        
        await ctx.info(f"Similar artist discovery completed. Found {total_found} unique recommendations")
        return response
        
    except Exception as e:
        logger.error(f"Failed to discover similar artists: {str(e)}")
        raise ValueError(f"Failed to discover similar artists: {str(e)}")

def main():
    """Run the Music Research MCP server."""
    logger.info("Starting Music Research MCP server")
    mcp.run()

if __name__ == "__main__":
    main()
