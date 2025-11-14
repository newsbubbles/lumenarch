"""Google Custom Search API Client for Empire.AI Music Video Maker

Provides access to Google's image and web search capabilities
for finding reference images and visual content.
"""

import os
from typing import Optional, List, Dict, Any, Literal
import httpx
from pydantic import BaseModel, Field


class SearchImage(BaseModel):
    """Google Custom Search image result."""
    url: str = Field(description="Image URL")
    title: Optional[str] = Field(None, description="Image title")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    size: Optional[str] = Field(None, description="Image size description")
    context_link: Optional[str] = Field(None, description="URL of page containing the image")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    thumbnail_width: Optional[int] = Field(None, description="Thumbnail width")
    thumbnail_height: Optional[int] = Field(None, description="Thumbnail height")


class SearchResult(BaseModel):
    """Google Custom Search result."""
    title: str = Field(description="Result title")
    link: str = Field(description="Result URL")
    snippet: Optional[str] = Field(None, description="Result snippet/description")
    display_link: Optional[str] = Field(None, description="Display URL")
    formatted_url: Optional[str] = Field(None, description="Formatted URL")
    image: Optional[SearchImage] = Field(None, description="Image information if this is an image result")


class SearchInfo(BaseModel):
    """Search metadata information."""
    search_time: float = Field(description="Search time in seconds")
    formatted_search_time: str = Field(description="Formatted search time")
    total_results: str = Field(description="Total results count")
    formatted_total_results: str = Field(description="Formatted total results")


class ImageSearchRequest(BaseModel):
    """Request for Google Custom Search image search."""
    query: str = Field(description="Search query")
    num: int = Field(10, description="Number of results to return (1-10)")
    start: int = Field(1, description="Index of first result (1-based)")
    safe: Literal["active", "off"] = Field("active", description="Safe search setting")
    img_size: Optional[Literal["huge", "icon", "large", "medium", "small", "xlarge", "xxlarge"]] = Field(
        None, description="Image size filter"
    )
    img_type: Optional[Literal["clipart", "face", "lineart", "stock", "photo", "animated"]] = Field(
        None, description="Image type filter"
    )
    img_color_type: Optional[Literal["color", "gray", "mono", "trans"]] = Field(
        None, description="Image color type filter"
    )
    img_dominant_color: Optional[Literal[
        "black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "teal", "white", "yellow"
    ]] = Field(None, description="Dominant color filter")
    rights: Optional[str] = Field(None, description="Usage rights filter")
    gl: Optional[str] = Field(None, description="Country/region code")
    hl: Optional[str] = Field(None, description="Language code")


class WebSearchRequest(BaseModel):
    """Request for Google Custom Search web search."""
    query: str = Field(description="Search query")
    num: int = Field(10, description="Number of results to return (1-10)")
    start: int = Field(1, description="Index of first result (1-based)")
    safe: Literal["active", "off"] = Field("active", description="Safe search setting")
    gl: Optional[str] = Field(None, description="Country/region code")
    hl: Optional[str] = Field(None, description="Language code")
    date_restrict: Optional[str] = Field(None, description="Date restriction (e.g., 'd1', 'w1', 'm1', 'y1')")
    site_search: Optional[str] = Field(None, description="Restrict search to specific site")
    site_search_filter: Optional[Literal["e", "i"]] = Field(None, description="Include (i) or exclude (e) site_search")


class SearchResponse(BaseModel):
    """Google Custom Search response."""
    items: List[SearchResult] = Field(description="Search results")
    search_information: SearchInfo = Field(description="Search metadata")
    queries: Dict[str, Any] = Field(description="Query information")
    context: Optional[Dict[str, Any]] = Field(None, description="Context information")
    spelling: Optional[Dict[str, Any]] = Field(None, description="Spelling suggestions")


class GoogleCustomSearchClient:
    """Google Custom Search API client for image and web search."""
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        """Initialize the Google Custom Search client.
        
        Args:
            api_key: Google API key. If not provided, will look for GOOGLE_API_KEY env var.
            search_engine_id: Custom Search Engine ID. If not provided, will look for GOOGLE_SEARCH_ENGINE_ID env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not self.api_key:
            raise ValueError(
                "Google API key is required. "
                "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        
        if not self.search_engine_id:
            raise ValueError(
                "Google Search Engine ID is required. "
                "Set GOOGLE_SEARCH_ENGINE_ID environment variable or pass search_engine_id parameter."
            )
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the Google Custom Search API."""
        request_params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            **params
        }
        
        response = await self.client.get(self.base_url, params=request_params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for Google API errors
        if "error" in data:
            error_info = data["error"]
            raise Exception(f"Google API error {error_info['code']}: {error_info['message']}")
        
        return data
    
    def _parse_search_result(self, item: Dict, is_image: bool = False) -> SearchResult:
        """Parse a search result item."""
        image_info = None
        
        if is_image and "image" in item:
            img_data = item["image"]
            image_info = SearchImage(
                url=item["link"],
                title=item.get("title"),
                width=img_data.get("width"),
                height=img_data.get("height"),
                size=img_data.get("size"),
                context_link=img_data.get("contextLink"),
                thumbnail_url=img_data.get("thumbnailLink"),
                thumbnail_width=img_data.get("thumbnailWidth"),
                thumbnail_height=img_data.get("thumbnailHeight")
            )
        
        return SearchResult(
            title=item.get("title", ""),
            link=item.get("link", ""),
            snippet=item.get("snippet"),
            display_link=item.get("displayLink"),
            formatted_url=item.get("formattedUrl"),
            image=image_info
        )
    
    def _parse_search_info(self, info: Dict) -> SearchInfo:
        """Parse search information."""
        return SearchInfo(
            search_time=float(info.get("searchTime", 0)),
            formatted_search_time=info.get("formattedSearchTime", "0"),
            total_results=info.get("totalResults", "0"),
            formatted_total_results=info.get("formattedTotalResults", "0")
        )
    
    async def search_images(self, request: ImageSearchRequest) -> SearchResponse:
        """Search for images using Google Custom Search.
        
        Args:
            request: Image search request parameters
            
        Returns:
            Search results containing images
        """
        params = {
            "q": request.query,
            "searchType": "image",
            "num": min(request.num, 10),
            "start": request.start,
            "safe": request.safe
        }
        
        # Add optional image filters
        if request.img_size:
            params["imgSize"] = request.img_size
        if request.img_type:
            params["imgType"] = request.img_type
        if request.img_color_type:
            params["imgColorType"] = request.img_color_type
        if request.img_dominant_color:
            params["imgDominantColor"] = request.img_dominant_color
        if request.rights:
            params["rights"] = request.rights
        if request.gl:
            params["gl"] = request.gl
        if request.hl:
            params["hl"] = request.hl
        
        data = await self._make_request(params)
        
        # Parse results
        items = []
        for item in data.get("items", []):
            result = self._parse_search_result(item, is_image=True)
            items.append(result)
        
        search_info = self._parse_search_info(data.get("searchInformation", {}))
        
        return SearchResponse(
            items=items,
            search_information=search_info,
            queries=data.get("queries", {}),
            context=data.get("context"),
            spelling=data.get("spelling")
        )
    
    async def search_web(self, request: WebSearchRequest) -> SearchResponse:
        """Search for web pages using Google Custom Search.
        
        Args:
            request: Web search request parameters
            
        Returns:
            Search results containing web pages
        """
        params = {
            "q": request.query,
            "num": min(request.num, 10),
            "start": request.start,
            "safe": request.safe
        }
        
        # Add optional filters
        if request.gl:
            params["gl"] = request.gl
        if request.hl:
            params["hl"] = request.hl
        if request.date_restrict:
            params["dateRestrict"] = request.date_restrict
        if request.site_search:
            params["siteSearch"] = request.site_search
        if request.site_search_filter:
            params["siteSearchFilter"] = request.site_search_filter
        
        data = await self._make_request(params)
        
        # Parse results
        items = []
        for item in data.get("items", []):
            result = self._parse_search_result(item, is_image=False)
            items.append(result)
        
        search_info = self._parse_search_info(data.get("searchInformation", {}))
        
        return SearchResponse(
            items=items,
            search_information=search_info,
            queries=data.get("queries", {}),
            context=data.get("context"),
            spelling=data.get("spelling")
        )
    
    async def search_artist_images(self, artist_name: str, era: Optional[str] = None, 
                                 style: Optional[str] = None, limit: int = 10) -> List[SearchImage]:
        """Search for artist images with optional era and style filters.
        
        Args:
            artist_name: Name of the artist
            era: Optional era specification (e.g., "1980s", "early career")
            style: Optional style specification (e.g., "concert", "portrait", "album cover")
            limit: Maximum number of images to return
            
        Returns:
            List of image results
        """
        query_parts = [artist_name]
        if era:
            query_parts.append(era)
        if style:
            query_parts.append(style)
        
        query = " ".join(query_parts)
        
        request = ImageSearchRequest(
            query=query,
            num=min(limit, 10),
            img_type="photo",
            safe="active"
        )
        
        response = await self.search_images(request)
        return [result.image for result in response.items if result.image]
    
    async def search_reference_images(self, concept: str, style: Optional[str] = None, 
                                    color_scheme: Optional[str] = None, limit: int = 10) -> List[SearchImage]:
        """Search for reference images for visual concepts.
        
        Args:
            concept: Visual concept to search for
            style: Optional style specification
            color_scheme: Optional color scheme
            limit: Maximum number of images to return
            
        Returns:
            List of image results
        """
        query_parts = [concept]
        if style:
            query_parts.append(style)
        if color_scheme:
            query_parts.append(color_scheme)
        
        query = " ".join(query_parts)
        
        request = ImageSearchRequest(
            query=query,
            num=min(limit, 10),
            img_type="photo",
            safe="active"
        )
        
        if color_scheme:
            # Try to map color scheme to dominant color
            color_mapping = {
                "warm": "orange",
                "cool": "blue",
                "monochrome": "gray",
                "vibrant": "red",
                "dark": "black",
                "light": "white"
            }
            if color_scheme.lower() in color_mapping:
                request.img_dominant_color = color_mapping[color_scheme.lower()]
        
        response = await self.search_images(request)
        return [result.image for result in response.items if result.image]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
