"""Offline Instagram ingestion for Creator Match AI."""


def __getattr__(name: str):
    if name == "download_influencer_last_photos":
        from ingest.download_instagram import download_influencer_last_photos

        return download_influencer_last_photos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["download_influencer_last_photos"]
