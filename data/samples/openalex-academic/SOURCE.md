# Sample: OpenAlex works (`openalex_academic_dump`)

Fifteen scholarly works from ONE part of the public OpenAlex works snapshot, whole and with every
column, as one parquet file in the snapshot's own layout. The family reads this directory with
`cyt source run openalex_academic_dump --from data/samples/openalex-academic --out STAGE`
(a directory laid out as the snapshot: `updated_date=YYYY-MM-DD/part_NNNN.parquet`).

## Origin

- Publisher: OpenAlex (OurResearch). Dataset: the OpenAlex works snapshot in parquet, public on
  `https://openalex.s3.amazonaws.com/data/parquet/works/` (documentation: <https://docs.openalex.org>).
  Licence: CC0 1.0, the dedication OpenAlex states for its data (`license.md` in its public
  documentation repository, `ourresearch/openalex-docs`). The abstracts and titles inside the records
  are third-party text that OpenAlex redistributes under that statement.
- Part: `updated_date=2026-02-09/part_0046.parquet`, 152,568,007 bytes, 159,767 works, sha256
  `220945f93be699cf538207a61e346a59fa1e0d5bc802f2aaa3c8636dba90a73a`. Read from the operator's
  preserved copy of the snapshot (the part named in `frontier.jsonl.gz`, whose `expected_bytes` is the
  size above); the same part is the one the 2026 pilot tranche materialised
  (`parent_identity.sha256` of that pack). The part itself is not in this repository.

## The file

`updated_date=2026-02-09/part_0046.parquet`: 155,515 bytes, sha256
`cbed9d118651ddf639672c199e509f449eba5e7a5076579b4c95451a24735961`. It holds 15 works in the part's own
order: the part's schema, every column, no value edited, shortened or removed. A parquet writer stamps its
own version in the footer, so another `pyarrow` writes other file bytes for the same rows. The stable
identity of a row is the sha256 of the JSON the extractor reads (DuckDB `to_json` of the whole row); the
table below gives it, and the cut tool prints it.

| role | works (OpenAlex id, language, year) |
|---|---|
| selected by the pinned policy and rendered (10) | W1043719763 (en, 2007), W114559381 (fr, 2010), W1433235132 (de, 2004), W1551380949 (no language, 2010), W2019609318 (en, 2013), W2062740319 (no language, 2012), W2266349331 (en, 2015), W2520232370 (fr, 2011), W2625258660 (it, 2017), W2905023637 (en, 2020) |
| selected, but the renderer refuses it (1): the old script writes it to its recovery stage | W3122717005 (en, 2007) |
| refused by the selection (4): medicine; no abstract; retracted; uncited, non-core venue | W153375954, W127391412, W6959737561, W1060887079 |

| work | sha256 of its JSON |
|---|---|
| W1043719763 | `31fac12f0da704a5cb76ec553c121a6bc4b552af91191d0763227d5df42de2b1` |
| W1060887079 | `2ca964fbae28e66a2dba4c4515e79b57d037b2af810908f26df9f1b8af07c3bb` |
| W114559381 | `dbf308355427451a1877bc8e5477b2ef1508d74c6758324ec94a13afdb232c87` |
| W127391412 | `422e4ca88eeaa2efd4eb3abccd25a257c7fb678949f3ffe5ec2136238f613ef8` |
| W1433235132 | `e3585dcf16665dcd63c3d4ae3d8f1fd381bc27e2e902971590974c831f71b93f` |
| W153375954 | `5851ba951c83ecc52266f17bd43fbec57a6f6c05cc0be2106144daa5ef2123d9` |
| W1551380949 | `42998013aa61a408b17ba0ee2e4ce741e602e65d481f45c420afe78c5d760f18` |
| W2019609318 | `82e584070cbea8519f188be97715c940535a43f8798dd9e2d237b1a8075bb0b5` |
| W2062740319 | `2b73b36313d04599322450ee898e96626fa27038adeb5f4e042d08ec669325d2` |
| W2266349331 | `4d212421e1d66f9bd2fcff66611178a959b8451f25211f4dc964ea818eb239b8` |
| W2520232370 | `eccc2abb0569462799c06f906b2c629c12c8a17db2d56fd7c6cca525e677a63e` |
| W2625258660 | `841d41222f4fdce3a85cfb7ed0fb0359580114a3c6ccaf729f04e2a9b1c175ff` |
| W2905023637 | `548c5239e6c786ac307d40523378626e09684a1c36fcf37a3e1c00f194048a42` |
| W3122717005 | `d1369f6fcd62c92032aed035648c0c79e95bc86ffe946ffca146e944d8368de1` |
| W6959737561 | `6a3b19bb0f360716ddc176a2796b6ec89ab6a475f37728176b12ec2314f6a776` |

The pinned selection admits 259 works of the whole part (258 render, 1 goes to recovery); the sample keeps
10 + 1 of them and 4 that it refuses, so a test sees both sides of every filter it holds.

## Persons

OpenAlex records the authors of each work (`authorships`: display name, OpenAlex author id, ORCID,
affiliations). They are names of natural persons, in the file because the rows are whole. The family does
not embed or ship them: they go only in the master-only part of a row (`_master.authors`, display names
only; no ORCID is read), which the compact dump strips (ADR-039).

## Plans and frontier (real, for the part-selection tests)

- `frontier.jsonl.gz`: the 2,216 rows of the snapshot acquisition's `frontier.jsonl` (part path, bytes,
  works kept by the pinned policy, works upstream). sha256 of the unzipped file
  `1f3a12f5628d80f8df87508af4ec3b23880266d70fd88874581340157a2cbff4`.
- `plans/*.PLAN.json.gz`: three selection plans the pipeline wrote against that frontier (one part;
  four parts; 386 parts in four partitions). The sha256 of each unzipped file is in the test module.
  The source-run sha256 they name is `b27163ed5bed84ce51289b690b21aeec655e77285449d5d4eab2b76fcad8b1cd`.

## How the cut was made

`tests/fixture_tools/cut_openalex.py PART.parquet OUT_DIR` copies the 15 works named in it, streaming by row
group (needs `pyarrow` and `duckdb`), and prints each work's sha256 as above. It is deterministic on the
rows. The works were picked to cover four languages and a missing one (en, fr, de, it, none), a DOI and no
DOI, one and several authors, an abstract of 105 characters and one of 2,700, and each filter of the
selection once. A test (`tests/test_aw_family_openalex_academic_dump.py`) checks the file, every row's JSON
hash and the cut tool against this note.
