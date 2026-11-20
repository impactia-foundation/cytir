# Sample: Service-Public fiches (`service_public_fiche`)

Six information sheets (`F*.xml`), whole and byte for byte as the publisher ships them. 51,754 bytes
in all. The family reads this directory with
`cyt source run service_public_fiche --from data/samples/service-public-fiche --out STAGE`.

## Origin

- Publisher: Direction de l'information legale et administrative (DILA), the service of the Prime
  Minister that edits Service-Public.fr.
- Dataset: "Vos droits et demarches" (particuliers), the open-data XML archive `vosdroits-latest.zip`
  published on data.gouv.fr (dataset catalogued under the Licence Ouverte).
- Archive downloaded on 2026-08-15; its sha256 is
  `31fd718f4f4fa2fde1a67afce584a1da04d60d6d17705c26c33fb9cb1fe18afc` (23,662,173 bytes, 5,551
  members). The archive itself is not in this repository.

## The six files

| member | bytes | sha256 |
|---|---:|---|
| `F1613.xml` | 6,338 | `edf1adb46f3e2afe60598518873c589052c7972f7ec55625da41be2e897475db` |
| `F2725.xml` | 6,271 | `9d6f8a7c568d7c9d15b5845e3b705d1e978debf9f0133955e9e6a8d5dfd68924` |
| `F13476.xml` | 6,925 | `35c89fd73300a65f2777465c72ac17b67383b60e4c374482f1f49b7b97c79cfb` |
| `F24415.xml` | 6,802 | `4a3ab0dfb9cb34852b444597f72c12e6ca3f4fa297b76e301d711ca3c2c7bd8c` |
| `F31417.xml` | 7,033 | `355564dccc834193529e0b46afd5f34f8f7b5533c4ce0bea57ce2c4720ba6844` |
| `F36491.xml` | 18,385 | `c8559967efa61df66311e37a5a4d4d37ec8fe65d45b5b48f3780e1329f292200` |

## How the cut was made

`tests/fixture_tools/cut_service_public_fiche.py ARCHIVE.zip OUT_DIR` copies the six named members
out of the archive, unmodified, in the order above (no re-encoding, no trimming). It is
deterministic: the same archive gives the same six files. The members were picked for being small
and for covering different topics (insurance, banking, family law, employment, housing). A test
(`tests/test_aw_family_service_public_fiche.py`) checks the sha256 of every file against this table.

## Personal data

A fiche is a public-service information sheet written by the publisher. Each of the six names an
institution as creator, publisher and contributor (the DILA) and no natural person; none holds an
e-mail address or a telephone number. Checked by reading the metadata fields and by search.

## Licence and attribution

Licence Ouverte / Open Licence version 2.0 (Etalab-2.0):
<https://www.etalab.gouv.fr/licence-ouverte-open-licence/>. The licence allows reuse, copy,
adaptation and redistribution, including commercial, on condition of naming the source and the date
of its last update. Attribution for these files:

> Source: Direction de l'information legale et administrative (DILA), Service-Public.fr, "Vos droits
> et demarches" (open-data XML archive), Licence Ouverte 2.0. Each file carries the date of its
> last update in its `dc:date` field (between 2025-02-06 and 2026-06-19).

Each index row the family makes carries the licence (`Etalab-2.0`); the owner and the licence version
are kept in the master-only part of the row and are removed from the redistributable dump.
