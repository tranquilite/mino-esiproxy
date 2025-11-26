"""
ESI API Abstraction Layer
Handles caching, ETag management, and rate limiting for EVE Online ESI API
"""
import os
import json
import requests
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import re

from rokh.cache_meta import CacheMetadata, RateLimitBucket, ErrorLimitTracker, EndpointConfig as EndpointConfigModel, init_db

# Define project root (goes up from ESIProxy/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent

# Configure logging
logger = logging.getLogger('ESIAbstraction')


class ESIAbstraction:
    """ESI API abstraction with caching and rate limit management."""

    ESI_BASE_URL = "https://esi.evetech.net/latest"
    CACHE_DIR = PROJECT_ROOT / "cache"

    def __init__(self, db_url: str = None, debug: bool = None):
        """
        Initialize the ESI abstraction.

        Args:
            db_url: Database URL (default: from DATABASE_URL env var or SQLite)
            debug: Enable debug output (default: from ESI_DEBUG env var or False)
        """
        self.CACHE_DIR.mkdir(exist_ok=True)
        (self.CACHE_DIR / "esi_endpoints").mkdir(exist_ok=True)

        # Set debug mode
        if debug is None:
            debug = os.getenv('ESI_DEBUG', 'false').lower() in ('true', '1', 'yes')
        self.debug = debug

        # Initialize database
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

        # Load ESI credentials if available
        self.client_id = os.getenv('ESI_CLIENT_ID')
        self.client_secret = os.getenv('ESI_CLIENT_SECRET')
        self.refresh_token = os.getenv('ESI_REFRESH_TOKEN')
        self.application_id = os.getenv('ESI_APPLICATION_ID')

    def _get_cache_path(self, endpoint: str) -> Path:
        """Generate cache file path for an endpoint."""
        # Sanitize endpoint for filesystem
        #cache_name = endpoint.replace('/', '_').replace('?', '_').strip('_')
        # return self.CACHE_DIR / f"{cache_name}.json"
        #return self.CACHE_DIR /  'esi_endpoints' / f"{cache_name}.json"

        # Extract category (first path segment)
        parts = endpoint.strip('/').split('/')
        category = parts[0] if parts else 'uncategorized'

        # Sanitize remaining path for filename
        cache_name = endpoint.replace('/', '_').replace('?', '_').strip('_')

        # Create category subdirectory
        category_dir = self.CACHE_DIR / 'esi_endpoints' / category
        category_dir.mkdir(parents=True, exist_ok=True)

        return category_dir / f"{cache_name}.json"

    def _get_endpoint_config(self, endpoint: str) -> dict:
        """
        Get endpoint configuration from database.

        Args:
            endpoint: The endpoint path (e.g., '/markets/prices' or '/loyalty/stores/1000035/offers')

        Returns:
            dict: Configuration with cache_duration_hours, description, etc.
        """
        db = self.SessionLocal()
        try:
            # Try exact match first
            config = db.query(EndpointConfigModel).filter(
                EndpointConfigModel.endpoint_pattern == endpoint,
                EndpointConfigModel.is_parameterized == 0
            ).first()

            if config:
                return {
                    'cache_duration_hours': config.cache_duration_hours,
                    'description': config.description,
                    'update_time_utc': config.update_time_utc
                }

            # Try pattern matching for parameterized endpoints
            parameterized_configs = db.query(EndpointConfigModel).filter(
                EndpointConfigModel.is_parameterized == 1
            ).all()

            for config in parameterized_configs:
                pattern_parts = config.endpoint_pattern.split('/')
                endpoint_parts = endpoint.split('/')

                if len(pattern_parts) == len(endpoint_parts):
                    match = True
                    for p, e in zip(pattern_parts, endpoint_parts):
                        if '{' in p:
                            # This is a parameter, accept any value
                            continue
                        elif p != e:
                            # Parts don't match
                            match = False
                            break

                    if match:
                        return {
                            'cache_duration_hours': config.cache_duration_hours,
                            'description': config.description,
                            'update_time_utc': config.update_time_utc
                        }

            # Return default config
            return {
                'cache_duration_hours': 1,
                'description': 'Default configuration',
                'update_time_utc': None
            }

        finally:
            db.close()

    def _get_cache_metadata(self, endpoint: str) -> CacheMetadata:
        """Get cache metadata from database."""
        db = self.SessionLocal()
        try:
            return db.query(CacheMetadata).filter(CacheMetadata.endpoint == endpoint).first()
        finally:
            db.close()

    def _load_cache(self, endpoint: str) -> dict:
        """Load cached response for an endpoint using database metadata."""
        metadata = self._get_cache_metadata(endpoint)
        if not metadata:
            return None

        cache_path = Path(metadata.cache_file)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return {
                'data': data,
                'etag': metadata.etag,
                'last_modified': metadata.last_modified,
                'expires': metadata.expires,
                'cached_at': metadata.cached_at.isoformat()
            }
        return None

    def _save_cache(self, endpoint: str, data: dict, headers: dict):
        """Save response to cache with metadata in database."""
        cache_path = self._get_cache_path(endpoint)

        # Save data to file
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save/update metadata in database
        db = self.SessionLocal()
        try:
            metadata = db.query(CacheMetadata).filter(CacheMetadata.endpoint == endpoint).first()

            if metadata:
                # Update existing metadata
                metadata.cache_file = str(cache_path)
                metadata.etag = headers.get('ETag')
                metadata.last_modified = headers.get('Last-Modified')
                metadata.expires = headers.get('Expires')
                metadata.cached_at = datetime.now(timezone.utc)
            else:
                # Create new metadata
                metadata = CacheMetadata(
                    endpoint=endpoint,
                    cache_file=str(cache_path),
                    etag=headers.get('ETag'),
                    last_modified=headers.get('Last-Modified'),
                    expires=headers.get('Expires'),
                    cached_at=datetime.now(timezone.utc)
                )
                db.add(metadata)

            db.commit()
        finally:
            db.close()

    def _is_cache_valid(self, cache_entry: dict, cache_duration_hours: int = 1) -> bool:
        """Check if cached data is still valid based on Last-Modified and cache duration."""
        if not cache_entry:
            return False

        last_modified = cache_entry.get('last_modified')
        if not last_modified:
            # No Last-Modified header, fall back to cached_at timestamp
            cached_at = datetime.fromisoformat(cache_entry.get('cached_at'))
            return datetime.now(timezone.utc) < cached_at + timedelta(hours=cache_duration_hours)

        # Parse Last-Modified header (RFC 2822 format)
        from email.utils import parsedate_to_datetime
        last_modified_dt = parsedate_to_datetime(last_modified)

        # Cache is valid if we haven't exceeded cache_duration_hours since last modification
        return datetime.now(timezone.utc) < last_modified_dt + timedelta(hours=cache_duration_hours)

    def _parse_rate_limit_window(self, limit_str: str) -> int:
        """
        Parse rate limit window from X-Ratelimit-Limit header.

        Args:
            limit_str: Format like "150/15m" or "600/1h"

        Returns:
            int: Window duration in seconds
        """
        match = re.match(r'(\d+)/(\d+)([mh])', limit_str)
        if not match:
            return 60  # Default to 60 seconds

        tokens, duration, unit = match.groups()
        duration = int(duration)

        if unit == 'm':
            return duration * 60
        elif unit == 'h':
            return duration * 3600
        return 60

    def _update_rate_limit(self, bucket_group: str, headers: dict, status_code: int):
        """
        Update rate limit tracking based on response headers.

        Args:
            bucket_group: The rate limit bucket group
            headers: Response headers
            status_code: HTTP status code to determine token consumption
        """
        # Determine tokens consumed based on status code
        if 200 <= status_code < 300:
            tokens_used = 2
        elif 300 <= status_code < 400:
            tokens_used = 1
        elif 400 <= status_code < 500:
            tokens_used = 5
        else:  # 5xx errors
            tokens_used = 0

        limit_str = headers.get('X-Ratelimit-Limit')
        remaining = headers.get('X-Ratelimit-Remaining')

        if not limit_str or remaining is None:
            return  # No rate limit headers, skip tracking

        # Parse limit
        limit_total = int(limit_str.split('/')[0])
        limit_window = self._parse_rate_limit_window(limit_str)
        remaining = int(remaining)

        # Update database
        db = self.SessionLocal()
        try:
            bucket = db.query(RateLimitBucket).filter(
                RateLimitBucket.bucket_group == bucket_group
            ).first()

            current_time = datetime.now(timezone.utc)

            if bucket:
                # Update existing bucket
                bucket.limit_total = limit_total
                bucket.limit_window_seconds = limit_window
                bucket.remaining_tokens = remaining
                bucket.last_updated = current_time

                # Update request history for floating window
                if bucket.request_history is None:
                    bucket.request_history = []

                # Add current request
                bucket.request_history.append({
                    'timestamp': current_time.isoformat(),
                    'tokens_used': tokens_used
                })

                # Clean old entries outside the window
                cutoff_time = current_time - timedelta(seconds=limit_window)
                bucket.request_history = [
                    entry for entry in bucket.request_history
                    if datetime.fromisoformat(entry['timestamp']) > cutoff_time
                ]
            else:
                # Create new bucket
                bucket = RateLimitBucket(
                    bucket_group=bucket_group,
                    limit_total=limit_total,
                    limit_window_seconds=limit_window,
                    remaining_tokens=remaining,
                    last_updated=current_time,
                    request_history=[{
                        'timestamp': current_time.isoformat(),
                        'tokens_used': tokens_used
                    }]
                )
                db.add(bucket)

            db.commit()
        finally:
            db.close()

    def _check_rate_limit(self, bucket_group: str, estimated_tokens: int = 2) -> tuple[bool, float]:
        """
        Check if we can make a request without hitting rate limit.

        Args:
            bucket_group: The rate limit bucket group
            estimated_tokens: Estimated tokens this request will consume (default: 2 for 2xx)

        Returns:
            tuple: (can_proceed: bool, wait_seconds: float)
        """
        db = self.SessionLocal()
        try:
            bucket = db.query(RateLimitBucket).filter(
                RateLimitBucket.bucket_group == bucket_group
            ).first()

            if not bucket:
                # No rate limit data yet, allow request
                return True, 0.0

            # Check if we have enough tokens
            if bucket.remaining_tokens >= estimated_tokens:
                return True, 0.0

            # Calculate when oldest request will expire from the window
            if bucket.request_history:
                oldest_entry = min(
                    bucket.request_history,
                    key=lambda x: x['timestamp']
                )
                oldest_time = datetime.fromisoformat(oldest_entry['timestamp'])
                window_end = oldest_time + timedelta(seconds=bucket.limit_window_seconds)
                wait_seconds = (window_end - datetime.now(timezone.utc)).total_seconds()

                logger.warning(f"Rate limit reached for bucket '{bucket_group}': {bucket.remaining_tokens}/{bucket.limit_total} tokens, waiting {wait_seconds:.2f}s")
                return False, max(0, wait_seconds)

            return True, 0.0
        finally:
            db.close()

    def _update_error_limit(self, headers: dict, status_code: int):
        """
        Update error limit tracking based on response headers.

        Args:
            headers: Response headers
            status_code: HTTP status code
        """
        error_remain = headers.get('X-ESI-Error-Limit-Remain')
        error_reset = headers.get('X-ESI-Error-Limit-Reset')
        is_error_limited = 1 if status_code == 420 else 0

        # Only update if we have error limit headers
        if error_remain is None and error_reset is None and status_code != 420:
            return

        db = self.SessionLocal()
        try:
            tracker = db.query(ErrorLimitTracker).filter(ErrorLimitTracker.id == 1).first()
            current_time = datetime.now(timezone.utc)

            if tracker:
                # Update existing tracker
                if error_remain is not None:
                    tracker.error_limit_remain = int(error_remain)
                if error_reset is not None:
                    tracker.error_limit_reset = int(error_reset)
                tracker.is_error_limited = is_error_limited
                tracker.last_updated = current_time
            else:
                # Create new tracker
                tracker = ErrorLimitTracker(
                    id=1,
                    error_limit_remain=int(error_remain) if error_remain else None,
                    error_limit_reset=int(error_reset) if error_reset else None,
                    is_error_limited=is_error_limited,
                    last_updated=current_time
                )
                db.add(tracker)

            db.commit()

            # Log error limit status
            if is_error_limited:
                logger.error(f"ERROR LIMITED (HTTP 420): Reset in {error_reset}s")
                if self.debug:
                    print(f"!!>  ERROR LIMITED (420): Reset in {error_reset}s")
            elif error_remain is not None:
                remain = int(error_remain)
                if remain < 20:
                    logger.warning(f"ERROR LIMIT CRITICAL: {remain} errors remaining before 420")
                    if self.debug:
                        print(f"!!>  ERROR LIMIT LOW: {remain} errors remaining before 420")
                elif remain < 50:
                    logger.info(f"Error limit: {remain} remaining")
                    if self.debug:
                        print(f"!!> Error limit: {remain} remaining")

        finally:
            db.close()

    def _check_error_limit(self) -> tuple[bool, float]:
        """
        Check if we are currently error-limited.

        Returns:
            tuple: (can_proceed: bool, wait_seconds: float)
        """
        db = self.SessionLocal()
        try:
            tracker = db.query(ErrorLimitTracker).filter(ErrorLimitTracker.id == 1).first()

            if not tracker:
                # No error limit data yet, allow request
                return True, 0.0

            # Check if we're error-limited
            if tracker.is_error_limited == 1:
                # Check if the reset time has passed
                time_since_update = (datetime.now(timezone.utc) - tracker.last_updated).total_seconds()

                if tracker.error_limit_reset and time_since_update < tracker.error_limit_reset:
                    # Still error-limited
                    wait_seconds = tracker.error_limit_reset - time_since_update
                    return False, max(0, wait_seconds)
                else:
                    # Reset time should have passed, allow request (it will update status)
                    return True, 0.0

            # Check if error limit is getting low (be cautious)
            if tracker.error_limit_remain is not None and tracker.error_limit_remain < 5:
                logger.warning(f"CRITICAL: Only {tracker.error_limit_remain} errors remaining before 420!")
                if self.debug:
                    print(f"!!>  CRITICAL: Only {tracker.error_limit_remain} errors remaining before 420!")

            return True, 0.0

        finally:
            db.close()

    def _check_remote_etag(self, endpoint: str, bucket_group: str = None) -> str:
        """
        Perform a HEAD request to get the current ETag from ESI.

        Args:
            endpoint: API endpoint path
            bucket_group: Rate limit bucket group (optional)

        Returns:
            str: Current ETag from remote server, or None if not available
        """
        url = f"{self.ESI_BASE_URL}{endpoint}"

        if self.debug:
            print(f"\n{'='*80}")
            print(f"HEAD REQUEST: {url}")
            print(f"{'='*80}")

        response = self.session.head(url)

        if self.debug:
            print(f"\nRESPONSE STATUS: {response.status_code}")
            print(f"RESPONSE HEADERS:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")
            print(f"{'='*80}\n")

        response.raise_for_status()

        # Track rate limit if bucket group provided
        if bucket_group:
            self._update_rate_limit(bucket_group, response.headers, response.status_code)

        # Track error limit
        self._update_error_limit(response.headers, response.status_code)

        return response.headers.get('ETag')

    def get(self, endpoint: str, cache_duration_hours: float = None, use_etag: bool = True,
            bucket_group: str = None, respect_rate_limit: bool = True, internal:bool = False) -> dict:
        """
        Make a GET request to ESI API with caching and rate limiting.

        Args:
            endpoint: API endpoint path (e.g., '/markets/prices')
            cache_duration_hours: How long to trust cached data (default: use endpoint config)
            use_etag: Whether to check ETag for freshness (default: True)
            bucket_group: Rate limit bucket group (optional, extracted from response if not provided)
            respect_rate_limit: Whether to respect rate limits and wait if necessary (default: True)

        Returns:
            dict: API response data
        """
        # Get endpoint-specific configuration from database
        endpoint_config = self._get_endpoint_config(endpoint)
        if cache_duration_hours is None:
            cache_duration_hours = endpoint_config['cache_duration_hours']

        # Check cache first
        if self.debug:
            print(f"\n🔍 Checking cache for: {endpoint}")
            print(f"   Config: {endpoint_config['description']}")
            print(f"   Cache duration: {cache_duration_hours} hours")
        cache_entry = self._load_cache(endpoint)

        if cache_entry and self._is_cache_valid(cache_entry, cache_duration_hours):
            # Cache is still valid based on time, return cached data
            if self.debug:
                print(f"> CACHE HIT: Returning cached data for {endpoint}")
                print(f"   Cached at: {cache_entry.get('cached_at')}")
                print(f"   ETag: {cache_entry.get('etag')}")
            return cache_entry['data']
        elif cache_entry:
            if self.debug:
                print(f"*> CACHE EXPIRED: Cache exists but expired for {endpoint}")
        else:
            if self.debug:
                print(f"!> CACHE MISS: No cache found for {endpoint}")

        # Cache expired or doesn't exist
        # If we have cached data and ETag support is enabled, check if remote has changed
        if cache_entry and use_etag and cache_entry.get('etag'):
            try:
                # Check error limit before making any requests
                can_proceed, wait_seconds = self._check_error_limit()
                if not can_proceed:
                    logger.warning(f"Error-limited, returning cached data for {endpoint}")
                    if self.debug:
                        print(f"!!>  ERROR LIMITED: Waiting {wait_seconds:.2f}s, returning cached data")
                    if cache_entry:
                        return cache_entry['data']
                    time.sleep(wait_seconds + 0.1)

                # Check rate limit before making HEAD request
                if bucket_group and respect_rate_limit:
                    can_proceed, wait_seconds = self._check_rate_limit(bucket_group, estimated_tokens=1)
                    if not can_proceed:
                        # Rate limited, but we have cache - return cached data
                        return cache_entry['data']

                remote_etag = self._check_remote_etag(endpoint, bucket_group)
                if remote_etag and remote_etag == cache_entry['etag']:
                    # Remote hasn't changed, update cache timestamp in database
                    db = self.SessionLocal()
                    try:
                        metadata = db.query(CacheMetadata).filter(CacheMetadata.endpoint == endpoint).first()
                        if metadata:
                            metadata.cached_at = datetime.now(timezone.utc)
                            db.commit()
                    finally:
                        db.close()
                    return cache_entry['data']
            except Exception:
                # If HEAD request fails, proceed with GET request
                pass

        # Either no cache, ETag changed, or HEAD request failed - fetch fresh data
        # Check error limit before making GET request
        can_proceed, wait_seconds = self._check_error_limit()
        if not can_proceed:
            logger.warning(f"Error-limited: waiting {wait_seconds:.2f}s for {endpoint}")
            if self.debug:
                print(f"!!>  ERROR LIMITED: Waiting {wait_seconds:.2f}s before making request")
            if cache_entry:
                if self.debug:
                    print(f"   Returning cached data instead")
                return cache_entry['data']
            time.sleep(wait_seconds + 0.1)

        # Check rate limit before making GET request
        if bucket_group and respect_rate_limit:
            can_proceed, wait_seconds = self._check_rate_limit(bucket_group, estimated_tokens=2)
            if not can_proceed:
                if wait_seconds > 0:
                    if self.debug:
                        print(f"\n*>  RATE LIMIT: Waiting {wait_seconds:.2f} seconds for bucket '{bucket_group}'")
                    # Wait for rate limit to clear
                    time.sleep(wait_seconds + 0.1)  # Add small buffer

        if internal is False:
            url = f"{self.ESI_BASE_URL}{endpoint}"
        else:
            url = endpoint

        if self.debug:
            print(f"\n{'='*80}")
            print(f"GET REQUEST: {url}")
            print(f"{'='*80}")

        response = self.session.get(url)

        if self.debug:
            print(f"\nRESPONSE STATUS: {response.status_code}")
            print(f"RESPONSE HEADERS:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")
            print(f"{'='*80}\n")

        # Extract bucket group from response if not provided
        if not bucket_group:
            bucket_group = response.headers.get('X-Ratelimit-Group')
            if bucket_group and self.debug:
                print(f"*>  Detected rate limit bucket: '{bucket_group}'")

        # Track rate limit
        if bucket_group:
            self._update_rate_limit(bucket_group, response.headers, response.status_code)
            if self.debug:
                print(f"*> Rate limit updated for bucket '{bucket_group}'")

        # Track error limit
        self._update_error_limit(response.headers, response.status_code)

        # Raise for HTTP errors
        response.raise_for_status()

        # Parse and cache new data
        data = response.json()
        self._save_cache(endpoint, data, response.headers)
        if self.debug:
            print(f"*> Cached response for {endpoint}")

        return data

    def get_paginated(self, endpoint: str, cache_duration_hours: float = None, use_etag: bool = True,
                     bucket_group: str = None, respect_rate_limit: bool = True) -> list:
        """
        Fetch all pages of a paginated endpoint and consolidate into a single list.

        This method automatically fetches all pages by:
        1. Making an initial request (with or without page param)
        2. Checking X-Pages header to determine total pages
        3. Fetching remaining pages sequentially
        4. Consolidating all results into a single list

        Args:
            endpoint: API endpoint path (e.g., '/markets/orders?region_id=10000002')
            cache_duration_hours: How long to trust cached data (default: use endpoint config)
            use_etag: Whether to check ETag for freshness (default: True)
            bucket_group: Rate limit bucket group (optional, extracted from response if not provided)
            respect_rate_limit: Whether to respect rate limits and wait if necessary (default: True)

        Returns:
            list: Consolidated data from all pages

        Example:
            # Fetch all market orders for a region
            orders = esi.get_paginated('/markets/orders?region_id=10000002')
        """
        if self.debug:
            print(f"\n*> Starting paginated fetch for: {endpoint}")

        # Determine if endpoint already has query params
        separator = '&' if '?' in endpoint else '?'

        # Track all results and last response for X-Pages header
        all_data = []
        total_pages = None

        # Fetch first page (or page already specified in endpoint)
        if 'page=' not in endpoint:
            first_endpoint = f"{endpoint}{separator}page=1"
        else:
            first_endpoint = endpoint

        # Make initial request and capture response to check X-Pages
        # We need to modify get() temporarily to return headers, or access them directly
        # For now, we'll make a direct request to get headers, then use cached result

        # Check error limit before making first page request
        can_proceed, wait_seconds = self._check_error_limit()
        if not can_proceed:
            if self.debug:
                print(f"!!>  ERROR LIMITED: Waiting {wait_seconds:.2f}s before paginated fetch")
            time.sleep(wait_seconds + 0.1)

        # Make the first request to get X-Pages header
        url = f"{self.ESI_BASE_URL}{first_endpoint}"

        if self.debug:
            print(f"\n{'='*80}")
            print(f"GET REQUEST (Page 1): {url}")
            print(f"{'='*80}")

        response = self.session.get(url)
        if self.debug:
            print(f"\nRESPONSE STATUS: {response.status_code}")
            print(f"RESPONSE HEADERS:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")

        # Extract total pages from X-Pages header
        total_pages = int(response.headers.get('X-Pages', 1))
        if self.debug:
            print(f"\n*> Total pages detected: {total_pages}")
            print(f"{'='*80}\n")

        # Track rate limit and save cache for first page
        if not bucket_group:
            bucket_group = response.headers.get('X-Ratelimit-Group')
            if bucket_group and self.debug:
                print(f"*>  Detected rate limit bucket: '{bucket_group}'")

        if bucket_group:
            self._update_rate_limit(bucket_group, response.headers, response.status_code)

        # Track error limit
        self._update_error_limit(response.headers, response.status_code)

        response.raise_for_status()

        first_page_data = response.json()
        self._save_cache(first_endpoint, first_page_data, response.headers)
        all_data.extend(first_page_data)

        if self.debug:
            print(f"> Page 1/{total_pages}: Fetched {len(first_page_data)} items")

        # Fetch remaining pages
        for page in range(2, total_pages + 1):
            # Build endpoint for this page
            if 'page=' in endpoint:
                # Replace existing page parameter
                import re
                page_endpoint = re.sub(r'page=\d+', f'page={page}', endpoint)
            else:
                page_endpoint = f"{endpoint}{separator}page={page}"

            if self.debug:
                print(f"\n📄 Fetching page {page}/{total_pages}...")

            # Use regular get() method for caching and rate limiting
            page_data = self.get(
                page_endpoint,
                cache_duration_hours=cache_duration_hours,
                use_etag=use_etag,
                bucket_group=bucket_group,
                respect_rate_limit=respect_rate_limit
            )

            all_data.extend(page_data)
            if self.debug:
                print(f"> Page {page}/{total_pages}: Fetched {len(page_data)} items (Total: {len(all_data)})")
        if self.debug:
            print(f"\n> PAGINATION COMPLETE: Fetched {len(all_data)} total items across {total_pages} pages\n")
        return all_data
