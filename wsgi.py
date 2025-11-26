"""
WSGI entry point for gunicorn
"""
from ESIProxy.service import app

if __name__ == "__main__":
    app.run()
