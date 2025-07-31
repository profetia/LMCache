import asyncio

from typing import Dict, List, Optional

from simm import kv

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj, MixedMemoryAllocator
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


def _get_key(key: CacheEngineKey) -> str:
    return str(hash(key))


def _get_metadata(memory_obj: MemoryObj) -> RemoteMetadata:
    return RemoteMetadata(
        memory_obj.get_size(),
        memory_obj.get_shape(),
        memory_obj.get_dtype(),
        memory_obj.get_memory_format(),
    )


class SimmConnector(RemoteConnector):

    def __init__(
        self,
        block_size: int,
        loop: asyncio.AbstractEventLoop,
        allocator: LocalCPUBackend,
    ) -> None:
        self.loop = loop
        self.allocator = allocator

        if hasattr(allocator.memory_allocator, "buffer"):
            self.mr_ext = kv.register_mr(allocator.memory_allocator.buffer)  # type: ignore
        elif isinstance(allocator.memory_allocator, MixedMemoryAllocator):
            self.mr_ext = kv.register_mr(
                allocator.memory_allocator.pin_allocator.buffer
            )
        else:
            raise ValueError(
                f"Unsupported allocator: {type(allocator.memory_allocator)}"
            )

        self.store = kv.Store(block_size)

        # TODO: store metadata different sized blocks in remote
        self.metas: Dict[str, RemoteMetadata] = {}

    async def exists(self, key: CacheEngineKey) -> bool:
        key_str = _get_key(key)
        return await self._exists_impl(key_str)
    
    async def _exists_impl(self, key_str: str) -> bool:
        return await self.store.exists_async(key_str)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = _get_key(key)
        meta = self.metas.get(key_str)
        if meta is None or meta.dtype is None:
            logger.debug("Unable to get: unsupported")
            return None

        memory_obj = self.allocator.allocate(meta.shape, meta.dtype, meta.fmt)
        if memory_obj is None or memory_obj.tensor is None:
            logger.debug(
                f"Unable to get: allocation failed: allocator={str(self.allocator)}"
            )
            return None

        block_view = kv.BlockView.from_tensor(memory_obj.tensor, self.mr_ext)
        ok = await self._get_impl(key_str, block_view)
        if not ok:
            logger.debug(f"Unable to get: not found: key={str(key)}, key_str={key_str}")
            return None

        return memory_obj
    
    async def _get_impl(self, key_str: str, block_view: kv.BlockView) -> bool:
        return await self.store.get_async(key_str, block_view)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = _get_key(key)
        meta = _get_metadata(memory_obj)
        self.metas[key_str] = meta

        block_view = kv.BlockView.from_tensor(memory_obj.tensor, self.mr_ext)
        ok = await self._put_impl(key_str, block_view)
        if not ok:
            logger.debug(
                f"Unable to put: already exist: key={str(key)}, key_str={key_str}"
            )
    
    async def _put_impl(self, key_str: str, block_view: kv.BlockView) -> bool:
        return await self.store.put_async(key_str, block_view)

    async def list(self) -> List[str]:  # type: ignore
        pass

    async def close(self):
        pass

class SimmConnectorSync(SimmConnector):

    async def _exists_impl(self, key_str: str) -> bool:
        ret = self.store.exists(key_str)
        return ret
    
    async def _get_impl(self, key_str: str, block_view: kv.BlockView) -> bool:
        return self.store.get(key_str, block_view)
    
    async def _put_impl(self, key_str: str, block_view: kv.BlockView) -> bool:
        return self.store.put(key_str, block_view)
