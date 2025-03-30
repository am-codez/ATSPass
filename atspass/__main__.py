"""
Main entry point for the ATSPass application.
"""

from .api.app import app

if __name__ == '__main__':
    app.run(debug=True, port=5000) 