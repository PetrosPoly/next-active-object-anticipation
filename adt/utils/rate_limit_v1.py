import threading
import time 
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncRateLimiter:
    def __init__(self, max_calls, period):
        self._max_calls = max_calls
        self._period = period
        self._calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            current = time.monotonic()
            logging.info(f"Current API calls in period: {len(self._calls)}")
            # Remove calls that are outside the time window
            while self._calls and self._calls[0] <= current - self._period:
                self._calls.pop(0)
            if len(self._calls) >= self._max_calls:
                sleep_time = self._period - (current - self._calls[0])
                await asyncio.sleep(sleep_time)
            self._calls.append(time.monotonic())

class AsyncRateLimiter_per_second:
    def __init__(self, max_calls_per_minute, max_calls_per_second):
        self._max_calls_per_minute = max_calls_per_minute
        self._max_calls_per_second = max_calls_per_second
        self._minute_calls = []
        self._second_calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            current = time.monotonic()
            # Clean up old calls
            self._minute_calls = [t for t in self._minute_calls if t > current - 60]
            self._second_calls = [t for t in self._second_calls if t > current - 1]
            logging.info(f"Current API calls per second: {len(self._second_calls)}")
            logging.info(f"Current API calls per minute: {len(self._minute_calls)}")
            # Enforce rate limits
            if (len(self._minute_calls) >= self._max_calls_per_minute or
                len(self._second_calls) >= self._max_calls_per_second):
                sleep_time = max(
                    (self._minute_calls[0] + 60 - current) if self._minute_calls else 0,
                    (self._second_calls[0] + 1 - current) if self._second_calls else 0
                )
                await asyncio.sleep(sleep_time)
            self._minute_calls.append(current)
            self._second_calls.append(current)

