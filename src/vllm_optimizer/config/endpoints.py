"""Format HTTP endpoints from validated host and port values."""


def http_endpoint(host: str, port: int) -> str:
    """Return an HTTP origin, bracketing IPv6 literals for URL parsing."""
    formatted = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{formatted}:{port}"
