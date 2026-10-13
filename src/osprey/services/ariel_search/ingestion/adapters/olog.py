"""Phoebus Olog ingestion adapter.

Reads entries from a `Phoebus Olog <https://github.com/Olog/phoebus-olog>`_
service over its REST API and stores them verbatim, the way the other
reference adapters store their logbooks. The adapter is **read-only**: it
reports no write support and refuses ``create_entry``, so it never publishes,
edits or deletes anything on the logbook.

How far back an ingest reaches is not the adapter's decision: pass ``since``
(``osprey ariel ingest --since``) or set ``ariel.ingestion.start_date`` to
bound it. With neither, the adapter starts at the logbook's own oldest entry.

Olog's search endpoint reads its ``start`` and ``end`` bounds as wall-clock
times in the server's own timezone and silently ignores other spellings, so
the adapter formats every window in the facility zone (``system.timezone``),
which is the zone a facility logbook server runs in. Attachment bytes sit
behind the service login, which this read-only ingest does not hold, so an
entry records its attachment names and count but no fetchable attachment.
"""

import asyncio
import ssl
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import aiohttp

from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.ingestion.base import FacilityAdapter
from osprey.services.ariel_search.models import EnhancedLogbookEntry
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.models import FacilityEntryCreateRequest

logger = get_logger("ariel")

#: The ``source_system`` value every row from this adapter carries.
OLOG_SOURCE_SYSTEM = "Olog"

#: Entries per request. Olog serves at most 1000 whatever ``size`` asks for.
OLOG_PAGE_SIZE = 1000

#: Elasticsearch's default ``max_result_window``: ``from + size`` past this is
#: refused, so a window holding more hits than this is split before paging.
OLOG_MAX_RESULT_WINDOW = 10000

#: The smallest window the adapter is willing to split to. A window this short
#: with more than :data:`OLOG_MAX_RESULT_WINDOW` entries is not a real logbook.
_MIN_WINDOW = timedelta(minutes=1)

#: The spelling Olog reads ``start``/``end`` in (server-local wall clock).
_OLOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _names_of(value: Any) -> list[str]:
    """Return the ``name`` of each dict in an Olog name list (logbooks, tags)."""
    if not isinstance(value, list):
        return []
    return [item["name"] for item in value if isinstance(item, dict) and item.get("name")]


class OlogAdapter(FacilityAdapter):
    """Read-only, verbatim adapter for a Phoebus Olog service.

    ``source_url`` is the service root (``https://olog.example.org/Olog``);
    the adapter appends ``/logs/search``. Entries are fetched in time windows
    of ``chunk_days``, each paged in :data:`OLOG_PAGE_SIZE` steps and split in
    half whenever it holds more hits than Elasticsearch will page through.
    """

    def __init__(self, config: "ARIELConfig") -> None:
        """Initialize the adapter."""
        super().__init__(config)

        if not config.ingestion or not config.ingestion.source_url:
            raise IngestionError(
                "source_url is required for the olog adapter: the Olog service root, "
                "e.g. https://olog.example.org/Olog",
                source_system=self.source_system_name,
            )
        source_url = config.ingestion.source_url.rstrip("/")
        if not source_url.startswith(("http://", "https://")):
            raise IngestionError(
                f"source_url must be an http(s) URL for the olog adapter, got {source_url!r}",
                source_system=self.source_system_name,
            )
        self.source_url = source_url
        self.search_url = f"{source_url}/logs/search"

        self.chunk_days = config.ingestion.chunk_days or 365
        self.request_timeout = config.ingestion.request_timeout_seconds or 60
        self.max_retries = config.ingestion.max_retries or 3
        self.retry_delay = config.ingestion.retry_delay_seconds or 5
        self.page_size = OLOG_PAGE_SIZE

    @property
    def source_system_name(self) -> str:
        """Return the source system identifier."""
        return OLOG_SOURCE_SYSTEM

    @property
    def supports_write(self) -> bool:
        """Never: this adapter reads the logbook and nothing else."""
        return False

    async def create_entry(self, request: "FacilityEntryCreateRequest") -> str:
        """Refuse: the Olog adapter is read-only.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "The Olog adapter is read-only: it never creates, edits or deletes logbook entries"
        )

    # ── Reading ──────────────────────────────────────────────────────────

    async def fetch_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[EnhancedLogbookEntry]:
        """Fetch entries from the Olog service, oldest window first.

        Args:
            since: Only fetch entries after this instant. When ``None``, the
                adapter starts at the logbook's oldest entry; bound an ingest
                with ``--since`` or ``ariel.ingestion.start_date`` instead.
            until: Only fetch entries before this instant (default: now).
            limit: Maximum number of entries to yield.

        Yields:
            EnhancedLogbookEntry objects, stored verbatim.
        """
        self.unreadable_entries = 0
        seen_ids: set[str] = set()
        count = 0
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)

        async with aiohttp.ClientSession(
            connector=self._create_connector(), timeout=timeout
        ) as session:
            ssl_context = self._ssl_context()
            start_date, end_date = await self._resolve_range(session, since, until, ssl_context)
            windows = self._generate_time_windows(start_date, end_date)
            logger.info(
                f"Fetching Olog entries from {start_date.isoformat()} to "
                f"{end_date.isoformat()} in {len(windows)} window(s)"
            )

            for window_start, window_end in windows:
                async for page in self._fetch_window(
                    session, window_start, window_end, ssl_context
                ):
                    for data in page:
                        entry_id = str(data.get("id", ""))
                        if not entry_id or entry_id in seen_ids:
                            continue
                        seen_ids.add(entry_id)

                        try:
                            entry = self._convert_entry(data)
                        except Exception as e:
                            self.unreadable_entries += 1
                            logger.warning(f"Failed to convert Olog entry {entry_id}: {e}")
                            continue

                        if not entry["raw_text"].strip():
                            continue
                        yield entry
                        count += 1
                        if limit and count >= limit:
                            logger.info(f"Reached limit of {limit} entries")
                            return

        logger.info(f"Fetched {count} entries from Olog")

    async def count_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int | None:
        """Count the entries a fetch over the same range would read.

        Returns:
            Olog's hit count for the range, or ``None`` when it is capped at
            :data:`OLOG_MAX_RESULT_WINDOW` and therefore not a count.
        """
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(
            connector=self._create_connector(), timeout=timeout
        ) as session:
            ssl_context = self._ssl_context()
            start_date, end_date = await self._resolve_range(session, since, until, ssl_context)
            page = await self._fetch_page_with_retry(
                session, start_date, end_date, 0, 1, ssl_context
            )
        hits = page.get("hitCount")
        if not isinstance(hits, int) or hits >= OLOG_MAX_RESULT_WINDOW:
            return None
        return hits

    async def _resolve_range(
        self,
        session: aiohttp.ClientSession,
        since: datetime | None,
        until: datetime | None,
        ssl_context: ssl.SSLContext | bool,
    ) -> tuple[datetime, datetime]:
        """Return the aware ``(start, end)`` a fetch covers.

        ``since`` is used as given; when it is ``None`` the start is the
        logbook's own oldest entry, found with one ``sort=up`` probe, so the
        adapter carries no hardcoded date of its own.
        """
        end_date = until or datetime.now(UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)
        if since is not None:
            start_date = since if since.tzinfo is not None else since.replace(tzinfo=UTC)
            return start_date, end_date
        start_date = await self._earliest_entry(session, ssl_context) or (
            end_date - timedelta(days=self.chunk_days)
        )
        return start_date, end_date

    async def _earliest_entry(
        self, session: aiohttp.ClientSession, ssl_context: ssl.SSLContext | bool
    ) -> datetime | None:
        """Return the oldest entry's time, or ``None`` if the probe finds none."""
        params = {"size": "1", "from": "0", "sort": "up"}
        try:
            async with session.get(self.search_url, params=params, ssl=ssl_context) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, ValueError) as e:
            logger.warning(f"Olog earliest-entry probe failed: {e}")
            return None
        logs = data.get("logs") if isinstance(data, dict) else None
        if not logs:
            return None
        try:
            return _parse_olog_time(logs[0].get("createdDate"))
        except ValueError:
            return None

    def _generate_time_windows(
        self, start_date: datetime, end_date: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Split ``[start_date, end_date)`` into ``chunk_days`` windows."""
        windows: list[tuple[datetime, datetime]] = []
        window_start = start_date
        while window_start < end_date:
            window_end = min(window_start + timedelta(days=self.chunk_days), end_date)
            windows.append((window_start, window_end))
            window_start = window_end
        return windows

    async def _fetch_window(
        self,
        session: aiohttp.ClientSession,
        window_start: datetime,
        window_end: datetime,
        ssl_context: ssl.SSLContext | bool,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Yield every page of raw entries in a window, splitting it as needed.

        The first page's ``hitCount`` says whether Elasticsearch can page the
        whole window; when it cannot, the window is halved and each half read
        the same way, down to :data:`_MIN_WINDOW`.
        """
        offset = 0
        while True:
            page = await self._fetch_page_with_retry(
                session, window_start, window_end, offset, self.page_size, ssl_context
            )
            hits = page.get("hitCount")
            if (
                offset == 0
                and isinstance(hits, int)
                and hits >= OLOG_MAX_RESULT_WINDOW
                and window_end - window_start > _MIN_WINDOW
            ):
                middle = window_start + (window_end - window_start) / 2
                logger.debug(
                    f"Olog window {window_start.isoformat()}-{window_end.isoformat()} holds "
                    f"{hits}+ entries; splitting at {middle.isoformat()}"
                )
                async for half in self._fetch_window(session, window_start, middle, ssl_context):
                    yield half
                async for half in self._fetch_window(session, middle, window_end, ssl_context):
                    yield half
                return

            logs = page.get("logs")
            if not isinstance(logs, list):
                logger.warning(
                    f"Olog returned no 'logs' list for window "
                    f"{window_start.isoformat()}-{window_end.isoformat()}"
                )
                return
            yield [data for data in logs if isinstance(data, dict)]
            offset += len(logs)
            if len(logs) < self.page_size or (isinstance(hits, int) and offset >= hits):
                return
            if offset >= OLOG_MAX_RESULT_WINDOW:
                logger.warning(
                    f"Olog window {window_start.isoformat()}-{window_end.isoformat()} "
                    f"stopped at {offset} entries: Elasticsearch pages no further"
                )
                return

    async def _fetch_page_with_retry(
        self,
        session: aiohttp.ClientSession,
        window_start: datetime,
        window_end: datetime,
        offset: int,
        size: int,
        ssl_context: ssl.SSLContext | bool,
    ) -> dict[str, Any]:
        """Fetch one search page with exponential backoff on transient failures.

        Raises:
            IngestionError: On a 4xx answer, or once ``max_retries`` is used up.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self._fetch_page(
                    session, window_start, window_end, offset, size, ssl_context
                )
            except aiohttp.ClientResponseError as e:
                if 400 <= e.status < 500:
                    raise IngestionError(
                        f"HTTP {e.status} error from Olog: {e.message}",
                        source_system=self.source_system_name,
                    ) from e
                last_error = e
            except (aiohttp.ClientError, TimeoutError) as e:
                last_error = e

            delay = self.retry_delay * (2**attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{self.max_retries} failed for Olog window "
                f"{window_start.isoformat()}-{window_end.isoformat()} offset {offset}, "
                f"retrying in {delay}s: {last_error}"
            )
            await asyncio.sleep(delay)

        raise IngestionError(
            f"Max retries ({self.max_retries}) exceeded fetching Olog: {last_error}",
            source_system=self.source_system_name,
        )

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        window_start: datetime,
        window_end: datetime,
        offset: int,
        size: int,
        ssl_context: ssl.SSLContext | bool,
    ) -> dict[str, Any]:
        """Fetch one page of ``/logs/search`` and return its JSON body."""
        params = self._search_params(window_start, window_end, offset, size)
        logger.debug(f"Fetching Olog page: {params}")
        async with session.get(self.search_url, params=params, ssl=ssl_context) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)
        if not isinstance(data, dict):
            raise IngestionError(
                f"Unexpected Olog search response type: {type(data).__name__}, expected object",
                source_system=self.source_system_name,
            )
        return data

    @staticmethod
    def _search_params(
        window_start: datetime, window_end: datetime, offset: int, size: int
    ) -> dict[str, str]:
        """Build the query for one page, with the bounds in the facility zone."""
        zone = get_facility_timezone()
        return {
            "start": window_start.astimezone(zone).strftime(_OLOG_TIME_FORMAT),
            "end": window_end.astimezone(zone).strftime(_OLOG_TIME_FORMAT),
            "from": str(offset),
            "size": str(size),
            "sort": "up",
        }

    # ── Converting ───────────────────────────────────────────────────────

    def _convert_entry(self, data: dict[str, Any]) -> EnhancedLogbookEntry:
        """Convert one Olog entry to an EnhancedLogbookEntry, verbatim.

        Raises:
            ValueError: If the entry has no readable ``createdDate``.
        """
        now = datetime.now(UTC)
        timestamp = _parse_olog_time(data.get("createdDate"))

        title = (data.get("title") or "").strip()
        body = (data.get("description") or data.get("source") or "").strip()
        raw_text = f"{title}\n\n{body}" if title and body else title or body

        metadata: dict[str, Any] = {}
        if title:
            metadata["title"] = title
        if data.get("level"):
            metadata["level"] = data["level"]
        logbooks = _names_of(data.get("logbooks"))
        if logbooks:
            metadata["logbooks"] = logbooks
        tags = _names_of(data.get("tags"))
        if tags:
            metadata["tags"] = tags
        modified = data.get("modifyDate")
        if modified is not None:
            try:
                metadata["modified"] = _parse_olog_time(modified).isoformat()
            except ValueError:
                pass

        # Every property, verbatim: Shift Info (operator, shift lead, phones,
        # emails, shift id), Log Entry Group, Client Info, and any other the
        # logbook carries, as {property: {attribute: value}}.
        properties: dict[str, dict[str, str]] = {}
        for prop in data.get("properties") or []:
            if not isinstance(prop, dict) or not prop.get("name"):
                continue
            attrs = {
                a["name"]: a.get("value", "")
                for a in prop.get("attributes") or []
                if isinstance(a, dict) and a.get("name")
            }
            if attrs:
                properties[str(prop["name"])] = attrs
        if properties:
            metadata["properties"] = properties

        attachments = data.get("attachments") or []
        filenames = [
            att["filename"]
            for att in attachments
            if isinstance(att, dict) and isinstance(att.get("filename"), str)
        ]
        if attachments:
            metadata["attachment_count"] = len(attachments)
        if filenames:
            metadata["attachment_filenames"] = filenames

        return {
            "entry_id": str(data["id"]),
            "source_system": self.source_system_name,
            "timestamp": timestamp,
            "author": data.get("owner") or "",
            "raw_text": raw_text,
            "attachments": [],
            "metadata": metadata,
            "created_at": now,
            "updated_at": now,
        }


def _parse_olog_time(value: object) -> datetime:
    """Read an Olog time (epoch milliseconds) as an aware UTC datetime.

    Raises:
        ValueError: If the value is missing or not an epoch.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Cannot parse Olog timestamp: {value!r}")
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError) as err:
        raise ValueError(f"Cannot parse Olog timestamp: {value!r}") from err
