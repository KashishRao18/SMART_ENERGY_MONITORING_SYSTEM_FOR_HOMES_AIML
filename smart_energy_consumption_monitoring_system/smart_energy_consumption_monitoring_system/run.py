"""
Smart Energy Consumption Monitoring System
Main entry point
"""

import logging
from app import app
from app.config import Config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting Smart Energy Consumption Monitoring System")
    logger.info("Environment: %s", Config.APP_ENV)
    logger.info("Debug: %s | Auto-reload: %s | Template reload: %s", Config.DEBUG, Config.DEBUG, Config.TEMPLATES_AUTO_RELOAD)
    logger.info("Access URL: http://localhost:%s", Config.PORT)
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT,
        use_reloader=Config.DEBUG,
    )
