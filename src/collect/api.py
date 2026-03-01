import os
import requests
import time


class BlueskyAPI:
    BASE_URL = "https://bsky.social/xrpc"

    def __init__(self, handle=None, app_password=None):
        self.handle = handle or os.getenv("BSKY_HANDLE")
        self.app_password = app_password or os.getenv("BSKY_APP_PASSWORD")
        self._token = None

    def _authenticate(self):
        url = f"{self.BASE_URL}/com.atproto.server.createSession"

        for _ in range(3):
            r = requests.post(
                url,
                json={
                    "identifier": self.handle,
                    "password": self.app_password
                }
            )

            if r.status_code == 429:
                time.sleep(1)
                continue

            r.raise_for_status()
            self._token = r.json()["accessJwt"]
            return

        raise RuntimeError("Failed to authenticate")

    @property
    def token(self):
        if not self._token:
            self._authenticate()
        return self._token

    def get(self, endpoint, params):
        headers = {"Authorization": f"Bearer {self.token}"}
        return requests.get(
            f"{self.BASE_URL}/{endpoint}",
            headers=headers,
            params=params
        )