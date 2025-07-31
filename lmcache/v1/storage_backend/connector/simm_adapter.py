import os

from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector


logger = init_logger(__name__)


class SimmConnectorAdapter(ConnectorAdapter):

    def __init__(self):
        super().__init__("simm://")

    def can_parse(self, url: str) -> bool:
        return url.startswith(self.schema)

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        from .simm_connector import SimmConnector, SimmConnectorSync

        parse_url = parse_remote_url(context.url)
        use_sync = int(parse_url.query_params.get("use_sync", ["0"])[0])

        server_list = os.getenv("SIMM_SERVER_LIST") or ""
        block_size = int(os.getenv("SIMM_KV_SERVICE_BLOCK_SIZE") or (1 << 20))
        logger.info(
            f"Creating Simm connector with: server_list={server_list}, block_size={block_size}"
        )

        if use_sync:
            logger.warning("Using synchronized version of Simm connector")
            return SimmConnectorSync(block_size, context.loop, context.local_cpu_backend)

        return SimmConnector(block_size, context.loop, context.local_cpu_backend)

