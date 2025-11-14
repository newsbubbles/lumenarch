"""Genius API Client for Empire.AI Music Video Maker."""

from .client import (
    GeniusClient,
    SearchRequest,
    SearchResponse,
    SongRequest,
    ArtistRequest,
    ArtistSongsRequest,
    ArtistSongsResponse,
    AnnotationRequest,
    GeniusArtist,
    GeniusAlbum,
    GeniusSong,
    GeniusAnnotation,
    GeniusReferent,
)

__all__ = [
    "GeniusClient",
    "SearchRequest",
    "SearchResponse",
    "SongRequest",
    "ArtistRequest",
    "ArtistSongsRequest",
    "ArtistSongsResponse",
    "AnnotationRequest",
    "GeniusArtist",
    "GeniusAlbum",
    "GeniusSong",
    "GeniusAnnotation",
    "GeniusReferent",
]
