# ctgov_psy sample (task 1407)

SYNTHETIC. Not a live capture. The JSON follows the field paths the adapter reads from the
ClinicalTrials.gov API v2 (`protocolSection.*`) and the EU CTIS public search (`data[]`); the NCT and CT
numbers are invented; the HTML page is invented text. Nothing was fetched.

- `ctg-page-1.json`, `ctg-page-2.json`: two pages of one query slice (page 1 carries `nextPageToken`;
  page 2 repeats one study to exercise URL deduplication; page 1 holds one study without an id).
- `ctis-page-1.json`: one search page (a psychiatric hit, a non-psychiatric ketamine-style hit, a hit
  without a number).
- `regulator-page.html`: one invented regulator page.
- `golden/records.jsonl`: the rows the driver produced BEFORE the move (origin/main at 0dffefb85), over
  these responses, with the language detector stubbed to "default language, not confident", the date
  fixed to 2026-10-30, and the slice/term/page lists cut to the first two slices, two CTIS terms and the
  fourth regulator page. Rows are sorted-key JSON, one per line.
