"""
server/app.py — OpenEnv required server entry point.

The OpenEnv validator expects:
  - A file at server/app.py
  - A callable main() function as the entry point
  - [project.scripts] server = "server.app:main"
"""

import uvicorn


def main():
    """Main entry point for the OpenEnv SQL Repair Environment server."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
