ARIEL reads a Phoebus Olog logbook through the new `olog` ingestion adapter.
The adapter is read-only (it never writes to the logbook) and stores entries
verbatim, like the other reference adapters, keeping the author, text and
shift properties. How far back an ingest reaches is set by `--since` or the
new `ariel.ingestion.start_date` config floor, not by the adapter.
