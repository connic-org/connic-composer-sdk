import ipaddress
import socket

import httpx
from markdownify import markdownify


def fetch_doc(url: str) -> str:
    """Fetch a document from a URL and return the markdownified text.

    Args:
        url (str): The URL of the document to fetch.

    Returns:
        str: The markdownified text of the document.
    """
    try:
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo:
            raise ValueError("URL must use HTTP(S), include a hostname, and exclude credentials")
        host = parsed.raw_host.decode("ascii")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = [ipaddress.ip_address(result[4][0]) for result in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
        if not addresses or any(
            not address.is_global or address.is_reserved or address.is_multicast or getattr(address, "is_site_local", False)
            for address in addresses
        ):
            raise ValueError("URL must resolve only to public IP addresses")
        with httpx.Client(follow_redirects=False, timeout=10, trust_env=False) as client:
            for index, address in enumerate(addresses):
                try:
                    response = client.get(
                        parsed.copy_with(host=str(address)),
                        headers={"Host": parsed.netloc.decode("ascii")},
                        extensions={"sni_hostname": host},
                    )
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if index == len(addresses) - 1:
                        raise
            response.raise_for_status()
            return markdownify(response.text)
    except (httpx.HTTPStatusError, httpx.RequestError, httpx.InvalidURL, OSError, ValueError) as e:
        return f"Encountered an HTTP error: {str(e)}"
