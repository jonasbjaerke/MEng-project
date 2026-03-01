# src/data/async_client.py

import asyncio
import aiohttp
import aiolimiter


class BlueskyAsyncClient:

    def __init__(self, rps=100):
        self.limiter = aiolimiter.AsyncLimiter(rps, 1)
        self.timeout = aiohttp.ClientTimeout(total=15)
        self.connector = aiohttp.TCPConnector(limit=300, limit_per_host=50)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def get(self, url, params=None, headers=None, retries=3):
        for attempt in range(retries):
            async with self.limiter:
                async with self.session.get(
                    url,
                    params=params,
                    headers=headers
                ) as r:

                    if r.status == 200:
                        return await r.json()

                    if 500 <= r.status < 600:
                        await asyncio.sleep(1)
                        continue

                    return None
        return None