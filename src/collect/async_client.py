import asyncio
import aiohttp
import aiolimiter


class BlueskyAsyncClient:
    """
    Central performance control for the entire pipeline.

    Tune only:
        rps
        concurrency
        timeout_total

    Everything else should remain unchanged.
    """

    def __init__(
        self,
        rps: int = 100,           # max requests per second
        concurrency: int = 100,   # max simultaneous in-flight requests
        timeout_total: int = 30  # total timeout per request
    ):

        # Throughput control
        self.limiter = aiolimiter.AsyncLimiter(rps, 1)
        self.semaphore = asyncio.Semaphore(concurrency)

        # Network timeout settings
        self.timeout = aiohttp.ClientTimeout(
            total=timeout_total,
            connect=10,
            sock_read=20
        )

        # Connection pool
        self.connector = aiohttp.TCPConnector(
            limit=concurrency,
            limit_per_host=concurrency,
            ttl_dns_cache=300
        )

        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()


    async def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        retries: int = 2
    ):
        """
        Safe GET request
        """

        for attempt in range(retries):

            try:
                async with self.semaphore:
                    async with self.limiter:
                        async with self.session.get(
                            url,
                            params=params,
                            headers=headers
                        ) as response:

                            # Success
                            if response.status == 200:
                                return await response.json()

                            # Rate limit
                            if response.status == 429:
                                await asyncio.sleep(2)
                                continue

                            # Server errors
                            if 500 <= response.status < 600:
                                await asyncio.sleep(1)
                                continue

                            # Other client errors
                            return None

            except (asyncio.TimeoutError, aiohttp.ClientError):
                if attempt == retries - 1:
                    return None
                await asyncio.sleep(1)

        return None