"""
EVE Online Image Cache
Handles caching of images from images.evetech.net for offline access
"""
import os
import requests
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from models.cache_meta import ImageCacheMetadata, init_db

# Define project root (goes up from ESIProxy/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from project root
load_dotenv(PROJECT_ROOT / ".env")

# Configure logging
logger = logging.getLogger('ImageCache')


class ImageCache:
    """Image cache for EVE Online image server (images.evetech.net)."""

    IMAGE_BASE_URL = "https://images.evetech.net"
    CACHE_DIR = PROJECT_ROOT / "cache"

    # Valid image endpoints and sizes (powers of 2 from 32 to 1024)
    # Structure: {category: {endpoint: [valid_sizes]}}
    VALID_ENDPOINTS = {
        'types': {
            'icon': [32, 64, 128, 256, 512, 1024],
            'render': [32, 64, 128, 256, 512, 1024]
        },
        'characters': {
            'portrait': [32, 64, 128, 256, 512, 1024]
        },
        'corporations': {
            'logo': [32, 64, 128, 256, 512, 1024]
        }
    }

    def __init__(self, db_url: str = None, debug: bool = None):
        """
        Initialize the image cache.

        Args:
            db_url: Database URL (default: from DATABASE_URL env var or SQLite)
            debug: Enable debug output (default: from ESI_DEBUG env var or False)
        """
        # Set debug mode
        if debug is None:
            debug = os.getenv('ESI_DEBUG', 'false').lower() in ('true', '1', 'yes')
        self.debug = debug

        # Initialize database (reuse same database as ESI cache)
        if db_url is None:
            db_url = os.getenv('DATABASE_URL', 'sqlite:///cache/esi_cache.db')
        self.engine, SessionLocal = init_db(db_url)
        self.SessionLocal = SessionLocal

        # Initialize HTTP session
        self.session = requests.Session()
        user_agent = os.getenv('ESI_PROXY_USER_AGENT', 'ESI-Proxy/1.0')
        self.session.headers.update({
            'User-Agent': user_agent
        })

        # Create cache directories
        self.CACHE_DIR.mkdir(exist_ok=True)
        for category in self.VALID_ENDPOINTS.keys():
            for endpoint in self.VALID_ENDPOINTS[category].keys():
                (self.CACHE_DIR / category / endpoint).mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, category: str, endpoint: str, entity_id: int, size: int) -> str:
        """
        Generate cache key for an image.

        Args:
            category: 'types', 'characters', or 'corporations'
            endpoint: 'icon', 'render', 'portrait', or 'logo'
            entity_id: EVE entity ID (type, character, corporation)
            size: Image size

        Returns:
            str: Cache key (e.g., 'types_icon_587_64', 'characters_portrait_91072482_64')
        """
        return f"{category}_{endpoint}_{entity_id}_{size}"

    def _get_cache_path(self, category: str, endpoint: str, entity_id: int, size: int, file_type: str = 'png') -> Path:
        """
        Generate cache file path for an image.

        Args:
            category: 'types', 'characters', or 'corporations'
            endpoint: 'icon', 'render', 'portrait', or 'logo'
            entity_id: EVE entity ID
            size: Image size
            file_type: File extension (png, jpg, etc.)

        Returns:
            Path: Full path to cache file (e.g., cache/types/icon/587_64.png)
        """
        return self.CACHE_DIR / category / endpoint / f"{entity_id}_{size}.{file_type}"

    def _get_image_metadata(self, cache_key: str) -> ImageCacheMetadata:
        """
        Get image metadata from database.

        Args:
            cache_key: Cache key for the image

        Returns:
            ImageCacheMetadata or None
        """
        db = self.SessionLocal()
        try:
            return db.query(ImageCacheMetadata).filter(
                ImageCacheMetadata.cache_key == cache_key
            ).first()
        finally:
            db.close()

    def _save_image_metadata(self, cache_key: str, category: str, endpoint: str, entity_id: int, size: int,
                            file_path: Path, file_type: str, content_length: int = None,
                            etag: str = None):
        """
        Save image metadata to database.

        Args:
            cache_key: Cache key for the image
            category: 'types', 'characters', or 'corporations'
            endpoint: 'icon', 'render', 'portrait', or 'logo'
            entity_id: EVE entity ID
            size: Image size
            file_path: Path to cached file
            file_type: File extension
            content_length: File size in bytes
            etag: ETag from server
        """
        db = self.SessionLocal()
        try:
            metadata = db.query(ImageCacheMetadata).filter(
                ImageCacheMetadata.cache_key == cache_key
            ).first()

            if metadata:
                # Update existing metadata
                metadata.file_path = str(file_path)
                metadata.file_type = file_type
                metadata.content_length = content_length
                metadata.etag = etag
                metadata.cached_at = datetime.now(timezone.utc)
            else:
                # Create new metadata
                metadata = ImageCacheMetadata(
                    cache_key=cache_key,
                    endpoint=endpoint,
                    type_id=entity_id,
                    size=size,
                    file_path=str(file_path),
                    file_type=file_type,
                    content_length=content_length,
                    etag=etag,
                    cached_at=datetime.now(timezone.utc)
                )
                db.add(metadata)

            db.commit()
        finally:
            db.close()

    def get_image(self, category: str, endpoint: str, entity_id: int, size: int = 64) -> Path:
        """
        Get an image from cache or download if not cached.

        Args:
            category: 'types', 'characters', or 'corporations'
            endpoint: 'icon', 'render', 'portrait', or 'logo'
            entity_id: EVE entity ID
            size: Image size (default: 64)

        Returns:
            Path: Path to cached image file

        Raises:
            ValueError: If category, endpoint or size is invalid
            requests.HTTPError: If image download fails

        Example:
            cache = ImageCache()
            # Get icon for Rifter (type ID 587)
            icon_path = cache.get_image('types', 'icon', 587, 64)
            # Get portrait for character
            portrait_path = cache.get_image('characters', 'portrait', 91072482, 64)
        """
        # Validate category
        if category not in self.VALID_ENDPOINTS:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {list(self.VALID_ENDPOINTS.keys())}")

        # Validate endpoint
        if endpoint not in self.VALID_ENDPOINTS[category]:
            raise ValueError(f"Invalid endpoint '{endpoint}' for category '{category}'. Valid endpoints: {list(self.VALID_ENDPOINTS[category].keys())}")

        # Validate size
        if size not in self.VALID_ENDPOINTS[category][endpoint]:
            raise ValueError(f"Invalid size {size} for endpoint '{category}/{endpoint}'. Valid sizes: {self.VALID_ENDPOINTS[category][endpoint]}")

        # Generate cache key
        cache_key = self._generate_cache_key(category, endpoint, entity_id, size)

        if self.debug:
            print(f"\n[IMAGE] Requesting image: {category}/{endpoint}/{entity_id} @ {size}px")

        # Check if already cached
        metadata = self._get_image_metadata(cache_key)
        if metadata:
            cache_path = Path(metadata.file_path)
            if cache_path.exists():
                if self.debug:
                    print(f"[CACHE HIT] {cache_path}")
                    print(f"   Cached at: {metadata.cached_at}")
                    print(f"   File size: {metadata.content_length} bytes")
                return cache_path
            else:
                if self.debug:
                    print(f"[WARNING] Cache entry exists but file missing: {cache_path}")

        # Not in cache, download from server
        url = f"{self.IMAGE_BASE_URL}/{category}/{entity_id}/{endpoint}?size={size}"

        if self.debug:
            print(f"\n{'='*80}")
            print(f"GET IMAGE: {url}")
            print(f"{'='*80}")

        response = self.session.get(url)

        if self.debug:
            print(f"\nRESPONSE STATUS: {response.status_code}")
            print(f"RESPONSE HEADERS:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")
            print(f"{'='*80}\n")

        response.raise_for_status()

        # Determine file type from Content-Type header
        content_type = response.headers.get('Content-Type', 'image/png')
        if 'png' in content_type:
            file_type = 'png'
        elif 'jpeg' in content_type or 'jpg' in content_type:
            file_type = 'jpg'
        else:
            # Default to png
            file_type = 'png'

        # Save image to cache
        cache_path = self._get_cache_path(category, endpoint, entity_id, size, file_type)
        with open(cache_path, 'wb') as f:
            f.write(response.content)

        # Save metadata
        content_length = len(response.content)
        etag = response.headers.get('ETag')
        self._save_image_metadata(
            cache_key=cache_key,
            category=category,
            endpoint=endpoint,
            entity_id=entity_id,
            size=size,
            file_path=cache_path,
            file_type=file_type,
            content_length=content_length,
            etag=etag
        )

        if self.debug:
            print(f"[CACHED] {cache_path}")
            print(f"   Size: {content_length} bytes")
            print(f"   Type: {file_type}")

        return cache_path

    def get_icon(self, type_id: int, size: int = 64) -> Path:
        """
        Get an icon image for a type.

        Args:
            type_id: EVE type ID
            size: Image size (default: 64, valid: 32, 64, 128, 256, 512, 1024)

        Returns:
            Path: Path to cached icon file

        Example:
            cache = ImageCache()
            icon = cache.get_icon(587, 64)  # Rifter icon at 64x64
        """
        return self.get_image('types', 'icon', type_id, size)

    def get_render(self, type_id: int, size: int = 512) -> Path:
        """
        Get a render image for a type (usually ships).

        Args:
            type_id: EVE type ID
            size: Image size (default: 512, valid: 32, 64, 128, 256, 512, 1024)

        Returns:
            Path: Path to cached render file

        Example:
            cache = ImageCache()
            render = cache.get_render(587, 512)  # Rifter render at 512x512
        """
        return self.get_image('types', 'render', type_id, size)

    def get_portrait(self, character_id: int, size: int = 64) -> Path:
        """
        Get a portrait image for a character.

        Args:
            character_id: EVE character ID
            size: Image size (default: 64, valid: 32, 64, 128, 256, 512, 1024)

        Returns:
            Path: Path to cached portrait file

        Example:
            cache = ImageCache()
            portrait = cache.get_portrait(91072482, 64)  # Character portrait at 64x64
        """
        return self.get_image('characters', 'portrait', character_id, size)

    def get_logo(self, corporation_id: int, size: int = 64) -> Path:
        """
        Get a logo image for a corporation.

        Args:
            corporation_id: EVE corporation ID
            size: Image size (default: 64, valid: 32, 64, 128, 256, 512, 1024)

        Returns:
            Path: Path to cached logo file

        Example:
            cache = ImageCache()
            logo = cache.get_logo(1686954550, 64)  # Corporation logo at 64x64
        """
        return self.get_image('corporations', 'logo', corporation_id, size)

    def is_cached(self, category: str, endpoint: str, entity_id: int, size: int) -> bool:
        """
        Check if an image is already cached.

        Args:
            category: 'types', 'characters', or 'corporations'
            endpoint: 'icon', 'render', 'portrait', or 'logo'
            entity_id: EVE entity ID
            size: Image size

        Returns:
            bool: True if image is cached, False otherwise
        """
        cache_key = self._generate_cache_key(category, endpoint, entity_id, size)
        metadata = self._get_image_metadata(cache_key)
        if metadata:
            cache_path = Path(metadata.file_path)
            return cache_path.exists()
        return False

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            dict: Statistics including total images, total size, breakdown by endpoint
        """
        db = self.SessionLocal()
        try:
            all_metadata = db.query(ImageCacheMetadata).all()

            stats = {
                'total_images': len(all_metadata),
                'total_size_bytes': sum(m.content_length or 0 for m in all_metadata),
                'by_endpoint': {},
                'by_size': {}
            }

            # Breakdown by endpoint
            for metadata in all_metadata:
                endpoint = metadata.endpoint
                if endpoint not in stats['by_endpoint']:
                    stats['by_endpoint'][endpoint] = {
                        'count': 0,
                        'size_bytes': 0
                    }
                stats['by_endpoint'][endpoint]['count'] += 1
                stats['by_endpoint'][endpoint]['size_bytes'] += metadata.content_length or 0

                # Breakdown by size
                size = metadata.size
                if size not in stats['by_size']:
                    stats['by_size'][size] = {
                        'count': 0,
                        'size_bytes': 0
                    }
                stats['by_size'][size]['count'] += 1
                stats['by_size'][size]['size_bytes'] += metadata.content_length or 0

            return stats
        finally:
            db.close()
