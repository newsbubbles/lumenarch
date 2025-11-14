"""Last.fm API Client for Empire.AI Music Video Maker

Provides access to Last.fm's music database for artist information,
biographies, tags, and similar artists.
"""

import os
from typing import Optional, List, Dict, Any
import httpx
from pydantic import BaseModel, Field


class LastFmImage(BaseModel):
    """Last.fm image in different sizes."""
    size: str = Field(description="Image size (small, medium, large, extralarge)")
    url: str = Field(description="Image URL")


class LastFmTag(BaseModel):
    """Last.fm tag information."""
    name: str = Field(description="Tag name")
    url: str = Field(description="Last.fm URL for the tag")


class LastFmSimilarArtist(BaseModel):
    """Similar artist information."""
    name: str = Field(description="Artist name")
    url: str = Field(description="Last.fm URL for the artist")
    images: List[LastFmImage] = Field(default_factory=list, description="Artist images")


class LastFmBiography(BaseModel):
    """Artist biography information."""
    published: Optional[str] = Field(None, description="Publication date")
    summary: str = Field(description="Biography summary (truncated at 300 chars)")
    content: str = Field(description="Full biography content")


class LastFmStats(BaseModel):
    """Artist statistics."""
    listeners: int = Field(description="Number of listeners")
    playcount: int = Field(description="Total play count")


class ArtistInfoRequest(BaseModel):
    """Request for artist information."""
    artist: Optional[str] = Field(None, description="Artist name (required unless mbid provided)")
    mbid: Optional[str] = Field(None, description="MusicBrainz ID (required unless artist provided)")
    lang: Optional[str] = Field(None, description="Language for biography (ISO 639 alpha-2 code)")
    autocorrect: bool = Field(True, description="Transform misspelled artist names")
    username: Optional[str] = Field(None, description="Include user's playcount for this artist")


class ArtistInfoResponse(BaseModel):
    """Artist information response."""
    name: str = Field(description="Artist name")
    mbid: Optional[str] = Field(None, description="MusicBrainz ID")
    url: str = Field(description="Last.fm URL for the artist")
    images: List[LastFmImage] = Field(default_factory=list, description="Artist images")
    streamable: bool = Field(description="Whether artist is streamable")
    stats: LastFmStats = Field(description="Artist statistics")
    similar: List[LastFmSimilarArtist] = Field(default_factory=list, description="Similar artists")
    tags: List[LastFmTag] = Field(default_factory=list, description="Artist tags")
    bio: LastFmBiography = Field(description="Artist biography")


class SimilarArtistsRequest(BaseModel):
    """Request for similar artists."""
    artist: Optional[str] = Field(None, description="Artist name (required unless mbid provided)")
    mbid: Optional[str] = Field(None, description="MusicBrainz ID (required unless artist provided)")
    limit: int = Field(50, description="Number of similar artists to return (max 100)")
    autocorrect: bool = Field(True, description="Transform misspelled artist names")


class SimilarArtistsResponse(BaseModel):
    """Similar artists response."""
    artists: List[LastFmSimilarArtist] = Field(description="List of similar artists")
    total: int = Field(description="Total number of similar artists available")


class TopTagsRequest(BaseModel):
    """Request for artist top tags."""
    artist: Optional[str] = Field(None, description="Artist name (required unless mbid provided)")
    mbid: Optional[str] = Field(None, description="MusicBrainz ID (required unless artist provided)")
    autocorrect: bool = Field(True, description="Transform misspelled artist names")


class TopTagsResponse(BaseModel):
    """Top tags response."""
    tags: List[LastFmTag] = Field(description="List of top tags for the artist")


class LastFmClient:
    """Last.fm API client for music research."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Last.fm client.
        
        Args:
            api_key: Last.fm API key. If not provided, will look for LASTFM_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("LASTFM_API_KEY")
        if not self.api_key:
            raise ValueError("Last.fm API key is required. Set LASTFM_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _make_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the Last.fm API."""
        request_params = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            **params
        }
        
        response = await self.client.get(self.base_url, params=request_params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for Last.fm API errors
        if "error" in data:
            raise Exception(f"Last.fm API error {data['error']}: {data.get('message', 'Unknown error')}")
        
        return data
    
    def _parse_images(self, images_data: List[Dict]) -> List[LastFmImage]:
        """Parse image data from Last.fm response."""
        images = []
        for img in images_data:
            if img.get("#text"):  # Only include images with actual URLs
                images.append(LastFmImage(
                    size=img.get("size", "unknown"),
                    url=img["#text"]
                ))
        return images
    
    def _parse_tags(self, tags_data: List[Dict]) -> List[LastFmTag]:
        """Parse tag data from Last.fm response."""
        tags = []
        for tag in tags_data:
            tags.append(LastFmTag(
                name=tag["name"],
                url=tag["url"]
            ))
        return tags
    
    def _parse_similar_artists(self, similar_data: List[Dict]) -> List[LastFmSimilarArtist]:
        """Parse similar artists data from Last.fm response."""
        artists = []
        for artist in similar_data:
            images = self._parse_images(artist.get("image", []))
            artists.append(LastFmSimilarArtist(
                name=artist["name"],
                url=artist["url"],
                images=images
            ))
        return artists
    
    async def get_artist_info(self, request: ArtistInfoRequest) -> ArtistInfoResponse:
        """Get detailed information about an artist.
        
        Args:
            request: Artist info request parameters
            
        Returns:
            Artist information including biography, stats, similar artists, and tags
        """
        if not request.artist and not request.mbid:
            raise ValueError("Either artist name or mbid must be provided")
        
        params = {}
        if request.artist:
            params["artist"] = request.artist
        if request.mbid:
            params["mbid"] = request.mbid
        if request.lang:
            params["lang"] = request.lang
        if request.autocorrect:
            params["autocorrect"] = "1"
        if request.username:
            params["username"] = request.username
        
        data = await self._make_request("artist.getinfo", params)
        artist_data = data["artist"]
        
        # Parse images
        images = self._parse_images(artist_data.get("image", []))
        
        # Parse similar artists
        similar_artists = []
        if "similar" in artist_data and "artist" in artist_data["similar"]:
            similar_artists = self._parse_similar_artists(artist_data["similar"]["artist"])
        
        # Parse tags
        tags = []
        if "tags" in artist_data and "tag" in artist_data["tags"]:
            tags = self._parse_tags(artist_data["tags"]["tag"])
        
        # Parse biography
        bio_data = artist_data.get("bio", {})
        bio = LastFmBiography(
            published=bio_data.get("published"),
            summary=bio_data.get("summary", ""),
            content=bio_data.get("content", "")
        )
        
        # Parse stats
        stats_data = artist_data.get("stats", {})
        stats = LastFmStats(
            listeners=int(stats_data.get("listeners", 0)),
            playcount=int(stats_data.get("playcount", 0))
        )
        
        return ArtistInfoResponse(
            name=artist_data["name"],
            mbid=artist_data.get("mbid"),
            url=artist_data["url"],
            images=images,
            streamable=artist_data.get("streamable") == "1",
            stats=stats,
            similar=similar_artists,
            tags=tags,
            bio=bio
        )
    
    async def get_similar_artists(self, request: SimilarArtistsRequest) -> SimilarArtistsResponse:
        """Get artists similar to the specified artist.
        
        Args:
            request: Similar artists request parameters
            
        Returns:
            List of similar artists with metadata
        """
        if not request.artist and not request.mbid:
            raise ValueError("Either artist name or mbid must be provided")
        
        params = {"limit": min(request.limit, 100)}  # Max limit is 100
        if request.artist:
            params["artist"] = request.artist
        if request.mbid:
            params["mbid"] = request.mbid
        if request.autocorrect:
            params["autocorrect"] = "1"
        
        data = await self._make_request("artist.getsimilar", params)
        similar_data = data["similarartists"]
        
        artists = self._parse_similar_artists(similar_data["artist"])
        
        return SimilarArtistsResponse(
            artists=artists,
            total=len(artists)
        )
    
    async def get_top_tags(self, request: TopTagsRequest) -> TopTagsResponse:
        """Get the top tags for an artist.
        
        Args:
            request: Top tags request parameters
            
        Returns:
            List of top tags for the artist
        """
        if not request.artist and not request.mbid:
            raise ValueError("Either artist name or mbid must be provided")
        
        params = {}
        if request.artist:
            params["artist"] = request.artist
        if request.mbid:
            params["mbid"] = request.mbid
        if request.autocorrect:
            params["autocorrect"] = "1"
        
        data = await self._make_request("artist.gettoptags", params)
        tags_data = data["toptags"]
        
        tags = self._parse_tags(tags_data.get("tag", []))
        
        return TopTagsResponse(tags=tags)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
