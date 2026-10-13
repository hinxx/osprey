"""Tests for the read-only, verbatim Phoebus Olog ingestion adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import aiohttp
import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import ConfigurationError, IngestionError
from osprey.services.ariel_search.ingestion.adapters import olog as olog_module
from osprey.services.ariel_search.ingestion.adapters.olog import (
    OLOG_MAX_RESULT_WINDOW,
    OlogAdapter,
)

STOCKHOLM = ZoneInfo("Europe/Stockholm")


def _make_config(**ingestion: Any) -> ARIELConfig:
    block = {
        "adapter": "olog",
        "source_url": "https://olog.example.org/Olog/",
        "max_retries": 2,
        "retry_delay_seconds": 0,
    }
    block.update(ingestion)
    return ARIELConfig.from_dict({"database": {"uri": "postgresql://test"}, "ingestion": block})


def _entry(entry_id: int, created: datetime, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": entry_id,
        "owner": "operator1",
        "title": f"Entry {entry_id}",
        "description": f"Body of entry {entry_id}.",
        "level": "Normal",
        "state": "Active",
        "createdDate": int(created.timestamp() * 1000),
        "modifyDate": None,
        "logbooks": [{"name": "Operations", "state": "Active"}],
        "tags": [],
        "properties": [],
        "attachments": [],
    }
    data.update(overrides)
    return data


SAMPLE_ENTRY = _entry(
    58301,
    datetime(2026, 10, 8, 21, 39, 40, 489000, tzinfo=UTC),
    owner="aron.almabni",
    title="RFQ conditioning updates",
    description="Interlocks bring P to 0 then back.\nPV TS2-010CRM:Vac-VGP-01100:PrsR at 22:14:13.",
    modifyDate=int(datetime(2026, 10, 9, tzinfo=UTC).timestamp() * 1000),
    tags=[{"name": "RFQ", "state": "Active"}],
    properties=[
        {
            "name": "Client Info",
            "attributes": [{"name": "Host Name", "value": "mcr-ws01.tn.example.org"}],
        },
        {
            "name": "Shift Info",
            "attributes": [
                {"name": "Operator", "value": "Aron Al Mabni"},
                {"name": "Operator Phone", "value": "+46700000011"},
                {"name": "Shift Lead Email", "value": "aron.almabni@example.org"},
                {"name": "Shift ID", "value": "20261010B"},
            ],
        },
        {
            "name": "Log Entry Group",
            "attributes": [{"name": "id", "value": "81fdd538-8fea-4470-8e7c-b37bb61ff851"}],
        },
    ],
    attachments=[
        {"id": "0f2b4df8", "filename": "68f902a6.png", "fileMetadataDescription": "image"}
    ],
)


class _FakeSession:
    """An aiohttp session whose ``get`` answers from a handler of the query params."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict[str, str]] = []

    def get(self, _url, params=None, **_kwargs):
        self.calls.append(dict(params))
        outcome = self.handler(params)
        response = AsyncMock()
        if isinstance(outcome, Exception):
            response.raise_for_status = MagicMock(side_effect=outcome)
        else:
            response.raise_for_status = MagicMock()
            response.json = AsyncMock(return_value=outcome)
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


def _paged(entries: list[dict[str, Any]], hit_count: int | None = None):
    """A handler serving ``entries`` by the ``from``/``size`` of each request."""

    def handler(params):
        offset, size = int(params["from"]), int(params["size"])
        return {
            "hitCount": len(entries) if hit_count is None else hit_count,
            "logs": entries[offset : offset + size],
        }

    return handler


async def _collect(adapter: OlogAdapter, **kwargs: Any) -> list[dict[str, Any]]:
    return [entry async for entry in adapter.fetch_entries(**kwargs)]


class TestRegistration:
    def test_framework_registers_olog(self):
        """The framework provider ships the adapter under the name config.yml uses."""
        from osprey.registry.builtins import FrameworkRegistryProvider

        adapters = {
            reg.name: reg
            for reg in FrameworkRegistryProvider().get_registry_config().ariel_ingestion_adapters
        }
        assert adapters["olog"].class_name == "OlogAdapter"
        assert adapters["olog"].module_path == olog_module.__name__


class TestConstruction:
    def test_source_url_required(self):
        with pytest.raises(IngestionError, match="source_url is required"):
            OlogAdapter(_make_config(source_url=None))

    def test_source_url_must_be_http(self):
        with pytest.raises(IngestionError, match="http"):
            OlogAdapter(_make_config(source_url="/data/olog.json"))

    def test_search_url_is_derived_from_the_service_root(self):
        adapter = OlogAdapter(_make_config())
        assert adapter.source_url == "https://olog.example.org/Olog"
        assert adapter.search_url == "https://olog.example.org/Olog/logs/search"
        assert adapter.source_system_name == "Olog"


class TestReadOnly:
    def test_never_supports_write(self):
        adapter = OlogAdapter(_make_config(write={"enabled": True, "write_url": "https://x"}))
        assert adapter.supports_write is False
        assert adapter.requires_write_auth is True

    @pytest.mark.asyncio
    async def test_create_entry_is_refused(self):
        adapter = OlogAdapter(_make_config())
        with pytest.raises(NotImplementedError, match="read-only"):
            await adapter.create_entry(MagicMock())


class TestConvertEntryVerbatim:
    def test_entry_is_stored_verbatim(self):
        adapter = OlogAdapter(_make_config())
        entry = adapter._convert_entry(SAMPLE_ENTRY)

        assert entry["entry_id"] == "58301"
        assert entry["source_system"] == "Olog"
        # Author kept, not dropped.
        assert entry["author"] == "aron.almabni"
        assert entry["timestamp"] == datetime(2026, 10, 8, 21, 39, 40, 489000, tzinfo=UTC)
        assert entry["attachments"] == []

        # Text kept exactly, no placeholders.
        assert entry["raw_text"] == (
            "RFQ conditioning updates\n\n"
            "Interlocks bring P to 0 then back.\n"
            "PV TS2-010CRM:Vac-VGP-01100:PrsR at 22:14:13."
        )
        assert "[name]" not in entry["raw_text"]
        assert "[phone]" not in entry["raw_text"]

        meta = entry["metadata"]
        assert meta["title"] == "RFQ conditioning updates"
        assert meta["level"] == "Normal"
        assert meta["logbooks"] == ["Operations"]
        assert meta["tags"] == ["RFQ"]
        assert meta["modified"] == "2026-10-09T00:00:00+00:00"
        assert meta["attachment_count"] == 1
        assert meta["attachment_filenames"] == ["68f902a6.png"]
        # Every property kept verbatim, including the shift contacts.
        assert meta["properties"]["Shift Info"] == {
            "Operator": "Aron Al Mabni",
            "Operator Phone": "+46700000011",
            "Shift Lead Email": "aron.almabni@example.org",
            "Shift ID": "20261010B",
        }
        assert meta["properties"]["Client Info"] == {"Host Name": "mcr-ws01.tn.example.org"}
        assert meta["properties"]["Log Entry Group"] == {
            "id": "81fdd538-8fea-4470-8e7c-b37bb61ff851"
        }

    def test_body_falls_back_to_source(self):
        adapter = OlogAdapter(_make_config())
        data = _entry(1, datetime(2024, 1, 1, tzinfo=UTC), description=None, source="**raw**")
        assert adapter._convert_entry(data)["raw_text"] == "Entry 1\n\n**raw**"

    def test_entry_with_no_properties_has_no_properties_key(self):
        adapter = OlogAdapter(_make_config())
        entry = adapter._convert_entry(_entry(1, datetime(2024, 1, 1, tzinfo=UTC)))
        assert "properties" not in entry["metadata"]
        assert entry["author"] == "operator1"

    def test_missing_created_date_is_unreadable(self):
        adapter = OlogAdapter(_make_config())
        with pytest.raises(ValueError, match="timestamp"):
            adapter._convert_entry(_entry(1, datetime(2024, 1, 1, tzinfo=UTC), createdDate=None))


class TestStartDateConfig:
    def test_start_date_parsed_to_utc(self):
        c = _make_config(start_date="2021-01-01")
        assert c.ingestion.start_date == datetime(2021, 1, 1, tzinfo=UTC)

    def test_no_start_date_is_none(self):
        assert _make_config().ingestion.start_date is None

    def test_bad_start_date_is_refused(self):
        with pytest.raises(ConfigurationError, match="start_date"):
            _make_config(start_date="not-a-date")


class TestRange:
    @pytest.mark.asyncio
    async def test_since_is_used_as_given(self):
        adapter = OlogAdapter(_make_config())
        session = _FakeSession(_paged([]))
        async with session as s:
            start, end = await adapter._resolve_range(
                s, datetime(2023, 3, 1, tzinfo=UTC), datetime(2023, 4, 1, tzinfo=UTC), False
            )
        assert start == datetime(2023, 3, 1, tzinfo=UTC)
        assert end == datetime(2023, 4, 1, tzinfo=UTC)
        # No probe when since is given.
        assert session.calls == []

    @pytest.mark.asyncio
    async def test_no_since_probes_the_oldest_entry(self):
        adapter = OlogAdapter(_make_config())
        oldest = datetime(2017, 12, 12, 9, 6, 53, tzinfo=UTC)
        session = _FakeSession(lambda params: {"logs": [_entry(1, oldest)]})
        async with session as s:
            start, _ = await adapter._resolve_range(
                s, None, datetime(2024, 1, 1, tzinfo=UTC), False
            )
        assert start == oldest
        assert session.calls[0]["sort"] == "up" and session.calls[0]["size"] == "1"

    def test_windows_follow_chunk_days(self):
        adapter = OlogAdapter(_make_config(chunk_days=10))
        start = datetime(2024, 1, 1, tzinfo=UTC)
        windows = adapter._generate_time_windows(start, start + timedelta(days=25))
        assert windows == [
            (start, start + timedelta(days=10)),
            (start + timedelta(days=10), start + timedelta(days=20)),
            (start + timedelta(days=20), start + timedelta(days=25)),
        ]

    def test_search_params_are_facility_wall_clock(self):
        with patch.object(olog_module, "get_facility_timezone", return_value=STOCKHOLM):
            params = OlogAdapter._search_params(
                datetime(2025, 9, 1, tzinfo=UTC), datetime(2025, 9, 3, tzinfo=UTC), 1000, 1000
            )
        assert params == {
            "start": "2025-09-01 02:00:00",
            "end": "2025-09-03 02:00:00",
            "from": "1000",
            "size": "1000",
            "sort": "up",
        }


class TestFetch:
    @pytest.mark.asyncio
    async def test_pages_through_a_window_and_deduplicates(self):
        adapter = OlogAdapter(_make_config(chunk_days=365))
        adapter.page_size = 2
        base = datetime(2024, 1, 1, tzinfo=UTC)
        entries = [_entry(i, base + timedelta(hours=i)) for i in range(1, 6)]
        entries.append(_entry(3, base + timedelta(hours=3)))  # repeated across pages
        session = _FakeSession(_paged(entries))

        with patch("aiohttp.ClientSession", return_value=session):
            got = await _collect(adapter, since=base, until=base + timedelta(days=1))

        assert [e["entry_id"] for e in got] == ["1", "2", "3", "4", "5"]
        assert [c["from"] for c in session.calls] == ["0", "2", "4"]
        assert adapter.unreadable_entries == 0

    @pytest.mark.asyncio
    async def test_limit_stops_early(self):
        adapter = OlogAdapter(_make_config())
        base = datetime(2024, 1, 1, tzinfo=UTC)
        session = _FakeSession(_paged([_entry(i, base) for i in range(1, 6)]))
        with patch("aiohttp.ClientSession", return_value=session):
            got = await _collect(adapter, since=base, until=base + timedelta(days=1), limit=3)
        assert len(got) == 3

    @pytest.mark.asyncio
    async def test_unreadable_entries_are_counted_and_skipped(self):
        adapter = OlogAdapter(_make_config())
        base = datetime(2024, 1, 1, tzinfo=UTC)
        session = _FakeSession(_paged([_entry(1, base, createdDate="yesterday"), _entry(2, base)]))
        with patch("aiohttp.ClientSession", return_value=session):
            got = await _collect(adapter, since=base, until=base + timedelta(days=1))
        assert [e["entry_id"] for e in got] == ["2"]
        assert adapter.unreadable_entries == 1

    @pytest.mark.asyncio
    async def test_window_over_the_result_cap_is_split(self):
        adapter = OlogAdapter(_make_config(chunk_days=2))
        base = datetime(2024, 1, 1, tzinfo=UTC)
        first = _entry(1, base + timedelta(hours=1))
        second = _entry(2, base + timedelta(hours=30))

        def handler(params):
            if params["start"] == "2024-01-01 00:00:00" and params["end"] == "2024-01-03 00:00:00":
                return {"hitCount": OLOG_MAX_RESULT_WINDOW, "logs": [first]}
            if params["end"] == "2024-01-02 00:00:00":
                return {"hitCount": 1, "logs": [first]}
            return {"hitCount": 1, "logs": [second]}

        session = _FakeSession(handler)
        with (
            patch.object(olog_module, "get_facility_timezone", return_value=UTC),
            patch("aiohttp.ClientSession", return_value=session),
        ):
            got = await _collect(adapter, since=base, until=base + timedelta(days=2))

        assert [e["entry_id"] for e in got] == ["1", "2"]
        assert [(c["start"], c["end"]) for c in session.calls] == [
            ("2024-01-01 00:00:00", "2024-01-03 00:00:00"),
            ("2024-01-01 00:00:00", "2024-01-02 00:00:00"),
            ("2024-01-02 00:00:00", "2024-01-03 00:00:00"),
        ]

    @pytest.mark.asyncio
    async def test_4xx_is_not_retried(self):
        adapter = OlogAdapter(_make_config())
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=401, message="Unauthorized"
        )
        session = _FakeSession(lambda params: error)
        base = datetime(2024, 1, 1, tzinfo=UTC)
        with (
            patch("aiohttp.ClientSession", return_value=session),
            pytest.raises(IngestionError, match="HTTP 401"),
        ):
            await _collect(adapter, since=base, until=base + timedelta(days=1))
        assert len(session.calls) == 1

    @pytest.mark.asyncio
    async def test_transient_failure_is_retried(self):
        adapter = OlogAdapter(_make_config(max_retries=3))
        base = datetime(2024, 1, 1, tzinfo=UTC)
        page = {"hitCount": 1, "logs": [_entry(1, base)]}
        session = _FakeSession(
            lambda params: aiohttp.ClientConnectionError("down") if not session.calls[1:] else page
        )
        with patch("aiohttp.ClientSession", return_value=session):
            got = await _collect(adapter, since=base, until=base + timedelta(days=1))
        assert [e["entry_id"] for e in got] == ["1"]
        assert len(session.calls) == 2


class TestCount:
    @pytest.mark.asyncio
    async def test_count_is_the_hit_count(self):
        adapter = OlogAdapter(_make_config())
        base = datetime(2024, 1, 1, tzinfo=UTC)
        session = _FakeSession(lambda params: {"hitCount": 42, "logs": []})
        with patch("aiohttp.ClientSession", return_value=session):
            assert await adapter.count_entries(since=base, until=base + timedelta(days=1)) == 42
        assert session.calls[-1]["size"] == "1"

    @pytest.mark.asyncio
    async def test_capped_hit_count_is_unknown(self):
        adapter = OlogAdapter(_make_config())
        base = datetime(2024, 1, 1, tzinfo=UTC)
        session = _FakeSession(lambda params: {"hitCount": OLOG_MAX_RESULT_WINDOW, "logs": []})
        with patch("aiohttp.ClientSession", return_value=session):
            assert await adapter.count_entries(since=base, until=base + timedelta(days=1)) is None
