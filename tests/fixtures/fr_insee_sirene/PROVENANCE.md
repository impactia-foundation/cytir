# fr_insee_sirene fixture (task 2728)

- Source: INSEE Base Sirene, StockUniteLegale parquet, via
  https://www.data.gouv.fr/api/1/datasets/r/350182c9-148a-46e0-8389-76c2ec1374a3
  (302 to static.data.gouv.fr/.../20261103-095402/stock-stockunitelegale-parquet.parquet, 717,824,037 B,
  last-modified 2026-11-03). Licence Ouverte / Etalab-2.0.
- Retrieved 2026-11-12 (UTC), by HTTP range GET (3.2 MB: footer + row group 0).
- `stock-unite-legale.slice.parquet`: 40 real rows of row group 0 (rows 1000-1039 of the subset
  statutDiffusion=O, etat=A, denomination present), all 35 source columns, rewritten with zstd.
  It is a SLICE, not the response body: the transport metadata hashes the slice.
- `.headers`: the real response headers of the redirect hop and the final 200 (HEAD, same day).
- `.transport.json`: `aw.http-fetch-metadata.v1`, written by hand around the slice (the real 717 MB body was not downloaded).
