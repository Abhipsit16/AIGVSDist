import argparse

from .config import Settings


def main():
    parser = argparse.ArgumentParser(description="AIGVS distributed service")
    parser.add_argument("command", choices=["serve", "migrate"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings()
    if args.command == "migrate":
        from .store import Store
        store = Store(settings)
        try:
            store.migrate()
        finally:
            store.close()
    else:
        import uvicorn
        uvicorn.run("aigvs.distributed.service:create_app", factory=True,
                    host=args.host, port=args.port, workers=1,
                    timeout_keep_alive=5, timeout_graceful_shutdown=60)


if __name__ == "__main__":
    main()
