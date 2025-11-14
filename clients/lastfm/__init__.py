"""Last.fm API Client for Empire.AI Music Video Maker."""

from .client import (
    LastFmClient,
    ArtistInfoRequest,
    ArtistInfoResponse,
    SimilarArtistsRequest,
    SimilarArtistsResponse,
    TopTagsRequest,
    TopTagsResponse,
    LastFmImage,
    LastFmTag,
    LastFmSimilarArtist,
    LastFmBiography,
    LastFmStats,
)

__all__ = [
    "LastFmClient",
    "ArtistInfoRequest",
    "ArtistInfoResponse", 
    "SimilarArtistsRequest",
    "SimilarArtistsResponse",
    "TopTagsRequest",
    "TopTagsResponse",
    "LastFmImage",
    "LastFmTag",
    "LastFmSimilarArtist",
    "LastFmBiography",
    "LastFmStats",
]
