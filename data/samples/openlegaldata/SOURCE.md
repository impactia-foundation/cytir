# Sample: OpenLegalData court decisions (`de_openlegaldata_v1`)

Five decisions in one parquet file laid out as the dump is: `original/parquet/dump-20260520/train-00000-of-00054.parquet`.
The family reads this directory with `cyt source run de_openlegaldata_v1 --from data/samples/openlegaldata --out STAGE`.

## Origin

- Publisher: OpenLegalData; dataset `openlegaldata/court-decisions-germany` on HuggingFace, revision `dump-20260520`
  (54 shards). Court decisions are official works without copyright (UrhG par. 5); the collection's metadata is ODbL,
  which the family never stamps (see the family's contract row).
- The five rows are REAL rows of shard 0 (`train-00000-of-00054.parquet`). They are not a new fetch: they come
  from the 28 real shard-0 rows that task 633 committed as the characterization sample of the materializer
  (the first 25 rows by id plus 3 real rows with an empty `markdown_content`), which the operator cut from the
  sealed acquisition copy of the shard (task 489). Nothing was read from the live source for this sample.
- The file is NOT the whole shard and its bytes are not the dump's bytes. It holds the columns the extractor reads
  (`id, slug, court, file_number, date, created_date, updated_date, type, ecli, content, markdown_content,
  reference_markers`; `court` as a struct) with every value as the sample rows carry it. A different `pyarrow`
  writes other file bytes for the same rows.

## The file

`train-00000-of-00054.parquet`: 26,048 bytes, sha256 `a1561afa83a6e4139dbfec431e88015e80a4479d50cb68d9b79525c8528f98e8`.
The sha256 above, and `source_part_sha256` in the fixture, is the hash of this re-encoded parquet, never the hash of the dump's shard 0.

| id | slug | role | sha256 of the row as JSON (sorted keys) |
|---|---|---|---|
| 58897 | bgh-2018-09-26-xii-za-1018 | rendered | `c050c9d60931da545991725cb43d7a41736d51d77c7edf75ff32a4c917a834e7` |
| 58910 | bgh-2018-09-25-x-zr-7618 | rendered | `4a61353b31683d1bc803821dc58b3640514393a8004de8cda30e872ff4c8a16d` |
| 58913 | bgh-2018-09-20-ix-zb-4116 | rendered | `dd48930461e254ee27c96a8e9b8af28ac8198d3edba98bb317cbfff13196ef12` |
| 58916 | bgh-2018-09-13-v-zb-23117 | rendered | `6aae78f5bac59e5ec1cbf52387e6fbf53ecee0a50aff444ea75df35dc7d7918b` |
| 59930 | bsg-2018-08-15-b-12-kr-817-r | real empty `markdown_content`: counted, not rendered | `2ae571c96b79c8de1c2f94b88b29533e3a7e56d9f99b2f966d5e6e2ac82a52cf` |

The four rendered rows are the shortest populated rows of the sample (3.5 to 7.3 thousand characters), so the
fixture stage stays small.
