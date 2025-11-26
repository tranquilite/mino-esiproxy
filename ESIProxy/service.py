"""
API Runner Service
"""
from flask import Flask, jsonify, request, send_file, Blueprint
from datetime import datetime, timezone, timedelta
from email.utils import formatdate
from dotenv import load_dotenv
from pathlib import Path

from .ESIAbstraction import ESIAbstraction
from .ImageCache import ImageCache
from . import extensions

# Load environment variables from project root
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

# Wakey, wakey, big mistakey
app = Flask(__name__)
esi = ESIAbstraction()
image_cache = ImageCache()

# Register extensions
for blueprint in extensions.modules:
    app.register_blueprint(blueprint)


@app.route('/markets/prices', methods=['GET'])
def get_market_prices():
    """
    Get market prices from ESI API with caching.
    """
    try:
        data = esi.get('/markets/prices')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/sovereignty/map', methods=['GET'])
def get_sovereignty_map():
    """
    Get sovereignty map from ESI API with caching and rate limiting.

    Rate limit: 600 tokens per 15 minutes (sovereignty group)
    Cache: Uses endpoint-specific configuration from endpoint_config.py
    """
    try:
        # Note: bucket_group will be auto-detected from X-Ratelimit-Group header
        data = esi.get(
            '/sovereignty/map',
            use_etag=True,
            respect_rate_limit=True
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/loyalty/stores/<int:corporation_id>/offers', methods=['GET'])
def get_loyalty_offers(corporation_id):
    """
    Get loyalty point offers for a corporation.

    Updates: Daily at 11:05 UTC
    """
    try:
        endpoint = f'/loyalty/stores/{corporation_id}/offers'
        data = esi.get(
            endpoint,
            use_etag=True,
            respect_rate_limit=True
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get('/corporations/npccorps')
def get_npc_corporations():
    """
    Get a list of npc corporations

    This route expires daily at 11:05
    Realistically though, this only updates whenever CCP releases new content.
    """
    try:
        endpoint = f'/corporations/npccorps'
        data = esi.get(
            endpoint,
            use_etag=True,
            respect_rate_limit=True
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get('/internal/loyalty/all')
def get_consolidated_loyalty_offers():
    """
    This one doesn't really bother to cache the result.
    Nevermind. Not caching was really annoying. Added caching.
    It just adds everything together, and the individual corps are cached anyway.
    """
    try:
        # Try to get from cache first
        consolidated_endpoint = '/internal/loyalty/all'
        try:
            cached_data = esi.get(
                consolidated_endpoint,
                cache_duration_hours=24,
                use_etag=False,
                respect_rate_limit=False,
                internal=True  # Mark as internal endpoint
            )
            return jsonify(cached_data)
        except:
            pass  # Cache miss or error, build it fresh
        # get all the npc corps
        endpoint = f'/corporations/npccorps'
        data = esi.get(
            endpoint,
            use_etag=True,
            respect_rate_limit=True
        )

        store_offers = {}

        for npc_corp_id in data:
            endpoint = f'/loyalty/stores/{npc_corp_id}/offers'
            loyalty_offers = esi.get(
                endpoint,
                use_etag=True,
                respect_rate_limit=True
            )

            if len(loyalty_offers) <= 0:  # if no loyalty store for corp
                continue

            store_offers[npc_corp_id] = loyalty_offers

        _now = datetime.now(timezone.utc)
        esi._save_cache(endpoint='/internal/loyalty/all', data=store_offers, headers={
            'Expires': formatdate((_now + timedelta(hours=24)).timestamp(), usegmt=True),
            'ETag': '',
            'Last-Modified': formatdate(_now.timestamp(), usegmt=True)
            }) 

        return jsonify(store_offers)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/industry/systems', methods=['GET'])
def get_industry_cost_indices():
    """
    Get industry cost indices for all solar systems.

    Updates: Hourly
    Cache: 1 hour (configured in database)
    """
    try:
        endpoint = '/industry/systems'
        data = esi.get(
            endpoint,
            use_etag=True,
            respect_rate_limit=True
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/markets/<int:region_id>/orders', methods=['GET'])
def get_region_market_orders(region_id):
    """
    Get all market orders for a region (paginated).

    Query Parameters:
        order_type (optional): Filter by order type (all, buy, sell). Default: all
        type_id (optional): Filter by item type ID

    Updates: Every 5 minutes
    Cache: 5 minutes per page (configured in database)

    Note: This endpoint is paginated. Uses get_paginated() to fetch all pages.

    Examples:
        /markets/10000002/orders
        /markets/10000002/orders?order_type=sell
        /markets/10000002/orders?type_id=34
        /markets/10000002/orders?order_type=buy&type_id=34
    """
    try:
        from flask import request

        # Build endpoint with query parameters
        endpoint = f'/markets/{region_id}/orders'

        # Build query string from request args
        query_params = []
        if request.args.get('order_type'):
            query_params.append(f"order_type={request.args.get('order_type')}")
        if request.args.get('type_id'):
            query_params.append(f"type_id={request.args.get('type_id')}")

        if query_params:
            endpoint += '?' + '&'.join(query_params)

        # This is a paginated endpoint - fetching all pages automatically
        data = esi.get_paginated(
            endpoint,
            use_etag=True,
            respect_rate_limit=True
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Image Cache Endpoints
# ============================================================================

@app.route('/types/<int:type_id>/icon', methods=['GET'])
def get_type_icon(type_id):
    """
    Get icon image for a type (drop-in replacement for images.evetech.net).

    Query Parameters:
        size (optional): Image size in pixels. Valid: 32, 64, 128, 256, 512, 1024. Default: 64
        tenant (optional): Ignored (for compatibility)

    Returns:
        PNG image file

    Examples:
        /types/587/icon              # Rifter icon at 64x64 (default)
        /types/587/icon?size=128     # Rifter icon at 128x128
    """
    try:
        size = int(request.args.get('size', 64))
        image_path = image_cache.get_icon(type_id, size)
        return send_file(image_path, mimetype='image/png')
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/types/<int:type_id>/render', methods=['GET'])
def get_type_render(type_id):
    """
    Get render image for a type (drop-in replacement for images.evetech.net).

    Query Parameters:
        size (optional): Image size in pixels. Valid: 32, 64, 128, 256, 512, 1024. Default: 512
        tenant (optional): Ignored (for compatibility)

    Returns:
        PNG/JPEG image file (format determined by Content-Type)

    Examples:
        /types/587/render            # Rifter render at 512x512 (default)
        /types/587/render?size=1024  # Rifter render at 1024x1024
    """
    try:
        size = int(request.args.get('size', 512))
        image_path = image_cache.get_render(type_id, size)
        # Determine mimetype from file extension
        mimetype = 'image/jpeg' if image_path.suffix == '.jpg' else 'image/png'
        return send_file(image_path, mimetype=mimetype)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/images/cache/status', methods=['GET'])
def get_image_cache_status():
    """
    Get image cache statistics.

    Returns:
        JSON with cache statistics including total images, size, and breakdowns

    Example:
        /images/cache/status
    """
    try:
        stats = image_cache.get_cache_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/characters/<int:character_id>/portrait', methods=['GET'])
def get_character_portrait(character_id):
    """
    Get portrait image for a character (drop-in replacement for images.evetech.net).

    Query Parameters:
        size (optional): Image size in pixels. Valid: 32, 64, 128, 256, 512, 1024. Default: 64
        tenant (optional): Ignored (for compatibility)

    Returns:
        PNG/JPEG image file

    Examples:
        /characters/91072482/portrait              # Portrait at 64x64 (default)
        /characters/91072482/portrait?size=128     # Portrait at 128x128
    """
    try:
        size = int(request.args.get('size', 64))
        image_path = image_cache.get_portrait(character_id, size)
        # Determine mimetype from file extension
        mimetype = 'image/jpeg' if image_path.suffix == '.jpg' else 'image/png'
        return send_file(image_path, mimetype=mimetype)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/corporations/<int:corporation_id>/logo', methods=['GET'])
def get_corporation_logo(corporation_id):
    """
    Get logo image for a corporation (drop-in replacement for images.evetech.net).

    Query Parameters:
        size (optional): Image size in pixels. Valid: 32, 64, 128, 256, 512, 1024. Default: 64
        tenant (optional): Ignored (for compatibility)

    Returns:
        PNG/JPEG image file

    Examples:
        /corporations/1686954550/logo              # Logo at 64x64 (default)
        /corporations/1686954550/logo?size=128     # Logo at 128x128
    """
    try:
        size = int(request.args.get('size', 64))
        image_path = image_cache.get_logo(corporation_id, size)
        # Determine mimetype from file extension
        mimetype = 'image/jpeg' if image_path.suffix == '.jpg' else 'image/png'
        return send_file(image_path, mimetype=mimetype)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/images/cache/check/<int:type_id>/<endpoint>', methods=['GET'])
def check_image_cached(type_id, endpoint):
    """
    Check if a specific image is cached.

    Path Parameters:
        type_id: EVE type ID
        endpoint: 'icon' or 'render'

    Query Parameters:
        size (optional): Image size. Default: 64 for icon, 512 for render

    Returns:
        JSON with cache status

    Examples:
        /images/cache/check/587/icon?size=64
        /images/cache/check/587/render?size=512
    """
    try:
        if endpoint == 'icon':
            default_size = 64
        elif endpoint == 'render':
            default_size = 512
        else:
            return jsonify({"error": "Invalid endpoint. Must be 'icon' or 'render'"}), 400

        size = int(request.args.get('size', default_size))
        is_cached = image_cache.is_cached('types', endpoint, type_id, size)

        return jsonify({
            "type_id": type_id,
            "endpoint": endpoint,
            "size": size,
            "is_cached": is_cached
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(error):
    return jsonify({}), 421


def main():
    """Main entry point for the API runner service."""
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
