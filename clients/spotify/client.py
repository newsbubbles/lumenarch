"""Spotify Web API Client for Empire.AI Music Video Maker

Provides access to Spotify's music database for artist information,
audio analysis, and music discovery.
"""

import os
import base64
from typing import Optional, List, Dict, Any, Literal
import httpx
from pydantic import BaseModel, Field


class SpotifyImage(BaseModel):
    """Spotify image information."""
    url: str = Field(description="Image URL")
    height: Optional[int] = Field(None, description="Image height in pixels")
    width: Optional[int] = Field(None, description="Image width in pixels")


class SpotifyExternalUrls(BaseModel):
    """External URLs for Spotify entities."""
    spotify: str = Field(description="Spotify URL")


class SpotifyFollowers(BaseModel):
    """Spotify followers information."""
    total: int = Field(description="Total number of followers")


class SpotifyArtistSimple(BaseModel):
    """Simplified artist information."""
    id: str = Field(description="Spotify artist ID")
    name: str = Field(description="Artist name")
    external_urls: SpotifyExternalUrls = Field(description="External URLs")
    href: str = Field(description="API endpoint for full artist object")
    uri: str = Field(description="Spotify URI")


class SpotifyArtistFull(SpotifyArtistSimple):
    """Full artist information."""
    followers: SpotifyFollowers = Field(description="Follower information")
    genres: List[str] = Field(description="Artist genres")
    images: List[SpotifyImage] = Field(description="Artist images")
    popularity: int = Field(description="Popularity score (0-100)")


class SpotifyAlbumSimple(BaseModel):
    """Simplified album information."""
    id: str = Field(description="Spotify album ID")
    name: str = Field(description="Album name")
    album_type: str = Field(description="Album type (album, single, compilation)")
    artists: List[SpotifyArtistSimple] = Field(description="Album artists")
    external_urls: SpotifyExternalUrls = Field(description="External URLs")
    href: str = Field(description="API endpoint for full album object")
    images: List[SpotifyImage] = Field(description="Album cover images")
    release_date: str = Field(description="Release date")
    release_date_precision: str = Field(description="Precision of release date (year, month, day)")
    total_tracks: int = Field(description="Total number of tracks")
    uri: str = Field(description="Spotify URI")


class SpotifyTrackSimple(BaseModel):
    """Simplified track information."""
    id: str = Field(description="Spotify track ID")
    name: str = Field(description="Track name")
    artists: List[SpotifyArtistSimple] = Field(description="Track artists")
    disc_number: int = Field(description="Disc number")
    duration_ms: int = Field(description="Track duration in milliseconds")
    explicit: bool = Field(description="Whether track has explicit lyrics")
    external_urls: SpotifyExternalUrls = Field(description="External URLs")
    href: str = Field(description="API endpoint for full track object")
    preview_url: Optional[str] = Field(None, description="Preview URL (30 seconds)")
    track_number: int = Field(description="Track number on album")
    uri: str = Field(description="Spotify URI")


class SpotifyTrackFull(SpotifyTrackSimple):
    """Full track information."""
    album: SpotifyAlbumSimple = Field(description="Album information")
    popularity: int = Field(description="Popularity score (0-100)")


class SpotifyAudioFeatures(BaseModel):
    """Audio features for a track."""
    id: str = Field(description="Spotify track ID")
    acousticness: float = Field(description="Acousticness (0.0-1.0)")
    danceability: float = Field(description="Danceability (0.0-1.0)")
    duration_ms: int = Field(description="Track duration in milliseconds")
    energy: float = Field(description="Energy (0.0-1.0)")
    instrumentalness: float = Field(description="Instrumentalness (0.0-1.0)")
    key: int = Field(description="Musical key (0-11, using Pitch Class notation)")
    liveness: float = Field(description="Liveness (0.0-1.0)")
    loudness: float = Field(description="Loudness in dB")
    mode: int = Field(description="Modality (0 = minor, 1 = major)")
    speechiness: float = Field(description="Speechiness (0.0-1.0)")
    tempo: float = Field(description="Tempo in BPM")
    time_signature: int = Field(description="Time signature (3-7)")
    valence: float = Field(description="Valence/positivity (0.0-1.0)")


class SearchRequest(BaseModel):
    """Request for searching Spotify content."""
    query: str = Field(description="Search query")
    search_type: List[Literal["artist", "album", "track"]] = Field(description="Types to search for")
    market: Optional[str] = Field(None, description="Market/country code (ISO 3166-1 alpha-2)")
    limit: int = Field(20, description="Number of results to return (1-50)")
    offset: int = Field(0, description="Index offset for pagination")


class SearchResponse(BaseModel):
    """Search results from Spotify."""
    artists: Optional[List[SpotifyArtistFull]] = Field(None, description="Artist search results")
    albums: Optional[List[SpotifyAlbumSimple]] = Field(None, description="Album search results")
    tracks: Optional[List[SpotifyTrackFull]] = Field(None, description="Track search results")


class ArtistRequest(BaseModel):
    """Request for artist information."""
    artist_id: str = Field(description="Spotify artist ID")


class ArtistAlbumsRequest(BaseModel):
    """Request for artist albums."""
    artist_id: str = Field(description="Spotify artist ID")
    include_groups: Optional[List[Literal["album", "single", "appears_on", "compilation"]]] = Field(
        None, description="Album types to include"
    )
    market: Optional[str] = Field(None, description="Market/country code")
    limit: int = Field(20, description="Number of results (1-50)")
    offset: int = Field(0, description="Index offset for pagination")


class ArtistTopTracksRequest(BaseModel):
    """Request for artist top tracks."""
    artist_id: str = Field(description="Spotify artist ID")
    market: str = Field("US", description="Market/country code (required)")


class RelatedArtistsRequest(BaseModel):
    """Request for related artists."""
    artist_id: str = Field(description="Spotify artist ID")


class AudioFeaturesRequest(BaseModel):
    """Request for audio features."""
    track_id: str = Field(description="Spotify track ID")


class SpotifyClient:
    """Spotify Web API client for music research."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """Initialize the Spotify client.
        
        Args:
            client_id: Spotify client ID. If not provided, will look for SPOTIFY_CLIENT_ID env var.
            client_secret: Spotify client secret. If not provided, will look for SPOTIFY_CLIENT_SECRET env var.
        """
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify client ID and secret are required. "
                "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables "
                "or pass client_id and client_secret parameters."
            )
        
        self.base_url = "https://api.spotify.com/v1"
        self.auth_url = "https://accounts.spotify.com/api/token"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[float] = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _get_access_token(self) -> str:
        """Get or refresh access token using client credentials flow."""
        import time
        
        # Check if we have a valid token
        if self.access_token and self.token_expires_at and time.time() < self.token_expires_at:
            return self.access_token
        
        # Get new token
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode("ascii")
        auth_b64 = base64.b64encode(auth_bytes).decode("ascii")
        
        headers = {
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {"grant_type": "client_credentials"}
        
        response = await self.client.post(self.auth_url, headers=headers, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data["access_token"]
        expires_in = token_data["expires_in"]
        self.token_expires_at = time.time() + expires_in - 60  # Refresh 60s early
        
        return self.access_token
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the Spotify API."""
        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        url = f"{self.base_url}{endpoint}"
        response = await self.client.get(url, headers=headers, params=params or {})
        response.raise_for_status()
        
        return response.json()
    
    def _parse_artist_simple(self, data: Dict) -> SpotifyArtistSimple:
        """Parse simple artist data."""
        return SpotifyArtistSimple(
            id=data["id"],
            name=data["name"],
            external_urls=SpotifyExternalUrls(**data["external_urls"]),
            href=data["href"],
            uri=data["uri"]
        )
    
    def _parse_artist_full(self, data: Dict) -> SpotifyArtistFull:
        """Parse full artist data."""
        images = [SpotifyImage(**img) for img in data["images"]]
        return SpotifyArtistFull(
            id=data["id"],
            name=data["name"],
            external_urls=SpotifyExternalUrls(**data["external_urls"]),
            href=data["href"],
            uri=data["uri"],
            followers=SpotifyFollowers(**data["followers"]),
            genres=data["genres"],
            images=images,
            popularity=data["popularity"]
        )
    
    def _parse_album_simple(self, data: Dict) -> SpotifyAlbumSimple:
        """Parse simple album data."""
        artists = [self._parse_artist_simple(artist) for artist in data["artists"]]
        images = [SpotifyImage(**img) for img in data["images"]]
        return SpotifyAlbumSimple(
            id=data["id"],
            name=data["name"],
            album_type=data["album_type"],
            artists=artists,
            external_urls=SpotifyExternalUrls(**data["external_urls"]),
            href=data["href"],
            images=images,
            release_date=data["release_date"],
            release_date_precision=data["release_date_precision"],
            total_tracks=data["total_tracks"],
            uri=data["uri"]
        )
    
    def _parse_track_full(self, data: Dict) -> SpotifyTrackFull:
        """Parse full track data."""
        artists = [self._parse_artist_simple(artist) for artist in data["artists"]]
        album = self._parse_album_simple(data["album"])
        return SpotifyTrackFull(
            id=data["id"],
            name=data["name"],
            artists=artists,
            disc_number=data["disc_number"],
            duration_ms=data["duration_ms"],
            explicit=data["explicit"],
            external_urls=SpotifyExternalUrls(**data["external_urls"]),
            href=data["href"],
            preview_url=data.get("preview_url"),
            track_number=data["track_number"],
            uri=data["uri"],
            album=album,
            popularity=data["popularity"]
        )
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search for artists, albums, or tracks.
        
        Args:
            request: Search request parameters
            
        Returns:
            Search results
        """
        params = {
            "q": request.query,
            "type": ",".join(request.search_type),
            "limit": min(request.limit, 50),
            "offset": request.offset
        }
        if request.market:
            params["market"] = request.market
        
        data = await self._make_request("/search", params)
        
        result = SearchResponse()
        
        if "artists" in data and "artist" in request.search_type:
            artists = [self._parse_artist_full(artist) for artist in data["artists"]["items"]]
            result.artists = artists
        
        if "albums" in data and "album" in request.search_type:
            albums = [self._parse_album_simple(album) for album in data["albums"]["items"]]
            result.albums = albums
        
        if "tracks" in data and "track" in request.search_type:
            tracks = [self._parse_track_full(track) for track in data["tracks"]["items"]]
            result.tracks = tracks
        
        return result
    
    async def get_artist(self, request: ArtistRequest) -> SpotifyArtistFull:
        """Get detailed information about an artist.
        
        Args:
            request: Artist request parameters
            
        Returns:
            Full artist information
        """
        data = await self._make_request(f"/artists/{request.artist_id}")
        return self._parse_artist_full(data)
    
    async def get_artist_albums(self, request: ArtistAlbumsRequest) -> List[SpotifyAlbumSimple]:
        """Get an artist's albums.
        
        Args:
            request: Artist albums request parameters
            
        Returns:
            List of artist albums
        """
        params = {
            "limit": min(request.limit, 50),
            "offset": request.offset
        }
        if request.include_groups:
            params["include_groups"] = ",".join(request.include_groups)
        if request.market:
            params["market"] = request.market
        
        data = await self._make_request(f"/artists/{request.artist_id}/albums", params)
        return [self._parse_album_simple(album) for album in data["items"]]
    
    async def get_artist_top_tracks(self, request: ArtistTopTracksRequest) -> List[SpotifyTrackFull]:
        """Get an artist's top tracks.
        
        Args:
            request: Artist top tracks request parameters
            
        Returns:
            List of top tracks
        """
        params = {"market": request.market}
        data = await self._make_request(f"/artists/{request.artist_id}/top-tracks", params)
        return [self._parse_track_full(track) for track in data["tracks"]]
    
    async def get_related_artists(self, request: RelatedArtistsRequest) -> List[SpotifyArtistFull]:
        """Get artists related to the specified artist.
        
        Args:
            request: Related artists request parameters
            
        Returns:
            List of related artists
        """
        data = await self._make_request(f"/artists/{request.artist_id}/related-artists")
        return [self._parse_artist_full(artist) for artist in data["artists"]]
    
    async def get_audio_features(self, request: AudioFeaturesRequest) -> SpotifyAudioFeatures:
        """Get audio features for a track.
        
        Args:
            request: Audio features request parameters
            
        Returns:
            Audio features for the track
        """
        data = await self._make_request(f"/audio-features/{request.track_id}")
        return SpotifyAudioFeatures(**data)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
