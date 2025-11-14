"""Genius API Client for Empire.AI Music Video Maker

Provides access to Genius's lyrics database and annotation system
for song lyrics and cultural context.
"""

import os
from typing import Optional, List, Dict, Any, Literal
import httpx
from pydantic import BaseModel, Field


class GeniusArtist(BaseModel):
    """Genius artist information."""
    id: int = Field(description="Genius artist ID")
    name: str = Field(description="Artist name")
    url: str = Field(description="Genius URL for the artist")
    image_url: Optional[str] = Field(None, description="Artist image URL")
    header_image_url: Optional[str] = Field(None, description="Artist header image URL")
    description: Optional[str] = Field(None, description="Artist description")
    followers_count: Optional[int] = Field(None, description="Number of followers")
    instagram_name: Optional[str] = Field(None, description="Instagram username")
    twitter_name: Optional[str] = Field(None, description="Twitter username")
    facebook_name: Optional[str] = Field(None, description="Facebook name")


class GeniusAlbum(BaseModel):
    """Genius album information."""
    id: int = Field(description="Genius album ID")
    name: str = Field(description="Album name")
    url: str = Field(description="Genius URL for the album")
    cover_art_url: Optional[str] = Field(None, description="Album cover art URL")
    release_date: Optional[str] = Field(None, description="Release date")
    artist: Optional[GeniusArtist] = Field(None, description="Primary artist")


class GeniusSong(BaseModel):
    """Genius song information."""
    id: int = Field(description="Genius song ID")
    title: str = Field(description="Song title")
    url: str = Field(description="Genius URL for the song")
    path: str = Field(description="URL path for the song")
    song_art_image_url: Optional[str] = Field(None, description="Song artwork URL")
    lyrics_owner_id: Optional[int] = Field(None, description="Lyrics owner ID")
    lyrics_state: Optional[str] = Field(None, description="Lyrics state (complete, incomplete, etc.)")
    release_date: Optional[str] = Field(None, description="Release date")
    featured_artists: List[GeniusArtist] = Field(default_factory=list, description="Featured artists")
    producer_artists: List[GeniusArtist] = Field(default_factory=list, description="Producer artists")
    writer_artists: List[GeniusArtist] = Field(default_factory=list, description="Writer artists")
    primary_artist: Optional[GeniusArtist] = Field(None, description="Primary artist")
    album: Optional[GeniusAlbum] = Field(None, description="Album information")
    description: Optional[str] = Field(None, description="Song description")
    lyrics: Optional[str] = Field(None, description="Full lyrics text")
    stats: Optional[Dict[str, Any]] = Field(None, description="Song statistics")


class GeniusAnnotation(BaseModel):
    """Genius annotation information."""
    id: int = Field(description="Annotation ID")
    body: str = Field(description="Annotation content")
    url: str = Field(description="Genius URL for the annotation")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Last update timestamp")
    votes_total: int = Field(description="Total votes")
    authors: List[Dict[str, Any]] = Field(default_factory=list, description="Annotation authors")


class GeniusReferent(BaseModel):
    """Genius referent (annotated text section)."""
    id: int = Field(description="Referent ID")
    annotator_id: Optional[int] = Field(None, description="Annotator ID")
    annotator_login: Optional[str] = Field(None, description="Annotator login")
    range: Dict[str, Any] = Field(description="Text range information")
    fragment: str = Field(description="Referenced text fragment")
    annotations: List[GeniusAnnotation] = Field(default_factory=list, description="Associated annotations")


class SearchRequest(BaseModel):
    """Request for searching Genius content."""
    query: str = Field(description="Search query")


class SearchResponse(BaseModel):
    """Search results from Genius."""
    songs: List[GeniusSong] = Field(description="Song search results")


class SongRequest(BaseModel):
    """Request for song information."""
    song_id: int = Field(description="Genius song ID")
    text_format: Literal["plain", "html", "dom"] = Field("html", description="Text format for lyrics")


class ArtistRequest(BaseModel):
    """Request for artist information."""
    artist_id: int = Field(description="Genius artist ID")
    text_format: Literal["plain", "html", "dom"] = Field("html", description="Text format for description")


class ArtistSongsRequest(BaseModel):
    """Request for artist songs."""
    artist_id: int = Field(description="Genius artist ID")
    sort: Literal["title", "popularity"] = Field("popularity", description="Sort order")
    page: int = Field(1, description="Page number")
    per_page: int = Field(20, description="Results per page (max 50)")


class ArtistSongsResponse(BaseModel):
    """Artist songs response."""
    songs: List[GeniusSong] = Field(description="List of artist songs")
    next_page: Optional[int] = Field(None, description="Next page number if available")


class AnnotationRequest(BaseModel):
    """Request for annotation information."""
    annotation_id: int = Field(description="Genius annotation ID")
    text_format: Literal["plain", "html", "dom"] = Field("html", description="Text format for content")


class GeniusClient:
    """Genius API client for lyrics and music knowledge."""
    
    def __init__(self, access_token: Optional[str] = None):
        """Initialize the Genius client.
        
        Args:
            access_token: Genius access token. If not provided, will look for GENIUS_ACCESS_TOKEN env var.
        """
        self.access_token = access_token or os.getenv("GENIUS_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError(
                "Genius access token is required. "
                "Set GENIUS_ACCESS_TOKEN environment variable or pass access_token parameter."
            )
        
        self.base_url = "https://api.genius.com"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the Genius API."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        url = f"{self.base_url}{endpoint}"
        response = await self.client.get(url, headers=headers, params=params or {})
        response.raise_for_status()
        
        data = response.json()
        
        # Check for Genius API errors
        if data.get("meta", {}).get("status") != 200:
            error_msg = data.get("meta", {}).get("message", "Unknown error")
            raise Exception(f"Genius API error: {error_msg}")
        
        return data["response"]
    
    def _parse_artist(self, data: Dict) -> GeniusArtist:
        """Parse artist data from Genius response."""
        return GeniusArtist(
            id=data["id"],
            name=data["name"],
            url=data["url"],
            image_url=data.get("image_url"),
            header_image_url=data.get("header_image_url"),
            description=data.get("description", {}).get("plain") if data.get("description") else None,
            followers_count=data.get("followers_count"),
            instagram_name=data.get("instagram_name"),
            twitter_name=data.get("twitter_name"),
            facebook_name=data.get("facebook_name")
        )
    
    def _parse_album(self, data: Optional[Dict]) -> Optional[GeniusAlbum]:
        """Parse album data from Genius response."""
        if not data:
            return None
        
        artist = None
        if data.get("artist"):
            artist = self._parse_artist(data["artist"])
        
        return GeniusAlbum(
            id=data["id"],
            name=data["name"],
            url=data["url"],
            cover_art_url=data.get("cover_art_url"),
            release_date=data.get("release_date_for_display"),
            artist=artist
        )
    
    def _parse_song(self, data: Dict, include_lyrics: bool = False) -> GeniusSong:
        """Parse song data from Genius response."""
        # Parse artists
        primary_artist = None
        if data.get("primary_artist"):
            primary_artist = self._parse_artist(data["primary_artist"])
        
        featured_artists = []
        if data.get("featured_artists"):
            featured_artists = [self._parse_artist(artist) for artist in data["featured_artists"]]
        
        producer_artists = []
        if data.get("producer_artists"):
            producer_artists = [self._parse_artist(artist) for artist in data["producer_artists"]]
        
        writer_artists = []
        if data.get("writer_artists"):
            writer_artists = [self._parse_artist(artist) for artist in data["writer_artists"]]
        
        # Parse album
        album = self._parse_album(data.get("album"))
        
        # Extract lyrics if available
        lyrics = None
        if include_lyrics and data.get("lyrics"):
            # Remove HTML tags for plain text lyrics
            import re
            lyrics_html = data["lyrics"]
            lyrics = re.sub(r'<[^>]+>', '', lyrics_html).strip()
        
        return GeniusSong(
            id=data["id"],
            title=data["title"],
            url=data["url"],
            path=data["path"],
            song_art_image_url=data.get("song_art_image_url"),
            lyrics_owner_id=data.get("lyrics_owner_id"),
            lyrics_state=data.get("lyrics_state"),
            release_date=data.get("release_date_for_display"),
            featured_artists=featured_artists,
            producer_artists=producer_artists,
            writer_artists=writer_artists,
            primary_artist=primary_artist,
            album=album,
            description=data.get("description", {}).get("plain") if data.get("description") else None,
            lyrics=lyrics,
            stats=data.get("stats")
        )
    
    def _parse_annotation(self, data: Dict) -> GeniusAnnotation:
        """Parse annotation data from Genius response."""
        return GeniusAnnotation(
            id=data["id"],
            body=data["body"]["plain"] if isinstance(data["body"], dict) else data["body"],
            url=data["url"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            votes_total=data["votes_total"],
            authors=data.get("authors", [])
        )
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search for songs on Genius.
        
        Args:
            request: Search request parameters
            
        Returns:
            Search results containing songs
        """
        params = {"q": request.query}
        data = await self._make_request("/search", params)
        
        songs = []
        for hit in data["hits"]:
            if hit["type"] == "song":
                song = self._parse_song(hit["result"])
                songs.append(song)
        
        return SearchResponse(songs=songs)
    
    async def get_song(self, request: SongRequest) -> GeniusSong:
        """Get detailed information about a song.
        
        Args:
            request: Song request parameters
            
        Returns:
            Detailed song information with lyrics
        """
        params = {"text_format": request.text_format}
        data = await self._make_request(f"/songs/{request.song_id}", params)
        
        return self._parse_song(data["song"], include_lyrics=True)
    
    async def get_artist(self, request: ArtistRequest) -> GeniusArtist:
        """Get detailed information about an artist.
        
        Args:
            request: Artist request parameters
            
        Returns:
            Detailed artist information
        """
        params = {"text_format": request.text_format}
        data = await self._make_request(f"/artists/{request.artist_id}", params)
        
        return self._parse_artist(data["artist"])
    
    async def get_artist_songs(self, request: ArtistSongsRequest) -> ArtistSongsResponse:
        """Get songs by an artist.
        
        Args:
            request: Artist songs request parameters
            
        Returns:
            List of artist songs with pagination info
        """
        params = {
            "sort": request.sort,
            "page": request.page,
            "per_page": min(request.per_page, 50)
        }
        data = await self._make_request(f"/artists/{request.artist_id}/songs", params)
        
        songs = [self._parse_song(song) for song in data["songs"]]
        
        # Check if there's a next page
        next_page = None
        if data.get("next_page"):
            next_page = request.page + 1
        
        return ArtistSongsResponse(songs=songs, next_page=next_page)
    
    async def get_annotation(self, request: AnnotationRequest) -> GeniusAnnotation:
        """Get detailed information about an annotation.
        
        Args:
            request: Annotation request parameters
            
        Returns:
            Detailed annotation information
        """
        params = {"text_format": request.text_format}
        data = await self._make_request(f"/annotations/{request.annotation_id}", params)
        
        return self._parse_annotation(data["annotation"])
    
    async def get_song_lyrics(self, song_id: int) -> Optional[str]:
        """Get clean lyrics text for a song.
        
        Args:
            song_id: Genius song ID
            
        Returns:
            Clean lyrics text without HTML formatting
        """
        try:
            song_request = SongRequest(song_id=song_id, text_format="plain")
            song = await self.get_song(song_request)
            return song.lyrics
        except Exception:
            return None
    
    async def search_song_by_title_artist(self, title: str, artist: str) -> Optional[GeniusSong]:
        """Search for a specific song by title and artist.
        
        Args:
            title: Song title
            artist: Artist name
            
        Returns:
            Best matching song or None if not found
        """
        search_query = f"{title} {artist}"
        search_request = SearchRequest(query=search_query)
        results = await self.search(search_request)
        
        # Find best match
        for song in results.songs:
            if (song.primary_artist and 
                artist.lower() in song.primary_artist.name.lower() and
                title.lower() in song.title.lower()):
                return song
        
        # Return first result if no exact match
        return results.songs[0] if results.songs else None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
