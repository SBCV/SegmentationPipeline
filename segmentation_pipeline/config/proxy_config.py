from pydantic import BaseModel


class ProxyConfig(BaseModel):
    ip_str: str = None
    port_str: str = None
