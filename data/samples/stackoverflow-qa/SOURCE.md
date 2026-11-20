# Sample: Stack Overflow questions and accepted answers (`stackoverflow_qa`)

Ten questions of Stack Overflow with their accepted answers, and four more rows the selection must
refuse: 24 `<row>` lines of the data dump's `Posts.xml`, whole and byte for byte as the dump ships
them, in the dump's own order, between its first two lines and its last line. 22,964 bytes in all.
The family reads this directory with
`cyt source run stackoverflow_qa --from data/samples/stackoverflow-qa --out STAGE`
(a directory holding the dump's `Posts.xml`; `Users.xml` too when a selected post names its owner by
id only; none of these does).

## Origin

- Publisher: Stack Exchange, Inc. Dataset: "Stack Exchange Data Dump", the public archive of the
  network's user-contributed content, kept at <https://archive.org/details/stackexchange>.
- File: `stackoverflow.com-Posts.7z`, 23,026,928,274 bytes, snapshot of 2024-04-07 (the archive's
  own mtime). Its sha256 is `1fcde86b9a0d701261a96698e78f65b7436d896eece5361ec7922d7c725c41cd`
  (the archive's own listing gives md5 `7fad535401201f725067170ee3b7bb87`, sha1
  `ea0b89fea1848593028a368b2bea046df9e835ef`). The archive itself is not in this repository.
- Read from the operator's preserved copy of the archive. The ten pairs are rows of the live
  tranches `stackoverflow-qa-score10-tranche-{000000-009999, 010000-019999, 020000-029999,
  030000-039999}`; the sha256 of each row below equals the sha256 that the sealed tranche receipt
  records for the same post (`raw_xml_sha256`), and the archive read for this cut is the one those
  tranches were made from.

## The file

`Posts.xml`: 22,964 bytes, sha256 `fa49210912c88cc2326762d9e5098f1296ae624ff49ca33949d5c75bb7e2a61a`. It
has the dump file's own first two lines (a byte-order mark and the XML declaration, then `<posts>`,
each ending in CR LF), the 24 rows below, and the closing `</posts>` as the archive ends. No row is
edited, shortened or re-encoded; no attribute is removed.

| question | its accepted answer | licence (question / answer, each row's own `ContentLicense`) |
|---|---|---|
| 1130479 | 1136143 | CC BY-SA 2.5 / 2.5 |
| 3283306 | 3283326 | 4.0 / 2.5 |
| 9560723 | 9815605 | 4.0 / 4.0 |
| 10269701 | 10269828 | 3.0 / 4.0 |
| 30676720 | 30803311 | 3.0 / 3.0 |
| 35666177 | 35666946 | 3.0 / 3.0 (the answer scores 0) |
| 40897069 | 40917149 | 3.0 / 3.0 |
| 42363405 | 42363636 | 4.0 / 3.0 (the answer scores -3) |
| 51081754 | 51102564 | 4.0 / 4.0 |
| 54220859 | 54220863 | 4.0 / 4.0 |

The family stamps the licence of the ACCEPTED ANSWER on the row (`CC-BY-SA-2.5`, `-3.0` or `-4.0`),
as the live rows do; the question's licence goes in the master-only part.

| negative row | what it is | why the selection refuses it |
|---|---|---|
| 38784 | a question with an accepted answer (41285, not in the sample) | score 7, below the bar of 10 |
| 10595 | a question with an accepted answer (11958, not in the sample) | score 0, below the bar of 10 |
| 1311 | a question with no accepted answer | score 21, but no accepted answer |
| 38834 | an answer that is not accepted (to question 38820) | not a question |

| row (post id) | sha256 of the row line |
|---|---|
| 1311 | `5759d1c59ecb8d81a48a58e90d0eff5786911eb33995e29603d2ff879e7f016c` |
| 10595 | `a3b8c264d7127af1f2b6e6d2ea4eb8112d6cc0136e56d13f0043bdaa6586e7ca` |
| 38784 | `016d059f009480d1a46f1691a636562b25bbcd621f3da16aefc44bb4f70c8224` |
| 38834 | `9d1025e207eb2edb83252d0ac6b6bcc63304269b22753ffe2c3eff6c9f9ed63e` |
| 1130479 | `60fad177d2da686e5610a7b0509e45cef87d0ec5d77ba2bdb12b30a0278b5eac` |
| 1136143 | `fe70b5673dcfa1fb4265e37d96cf9c99742a503ac90309f714db194f17647794` |
| 3283306 | `a327fd179afdc831d94c9b5f43538d01c5ce0f6a3b42bde3466d39dfd135ad96` |
| 3283326 | `118ab556785b80b3c844c97a3e5b2268874184ab84636f9efe6213965ae63a37` |
| 9560723 | `005462167c2662b06da1a7aa559e8461f6ceb1ef21dd5003a395da5d40f741cd` |
| 9815605 | `54cea17b8666de10603053c114630e75607391197dee724e2fce725c97223f15` |
| 10269701 | `0f009d490a42be9bbeb026a83ff7794d80fa3abf3d5335f0044adefc83b7e320` |
| 10269828 | `f22b5ea550cf357007d81f5c9220cad1930c870f53cdf1f4ebd9c4f74fc443cb` |
| 30676720 | `46baf89bf42289925d86ad79783cad4c896d202746f2d88564cd7cbf01815c84` |
| 30803311 | `a9f8f18e6f17168948ed709182c7006c6796f42be788b84a4ac4e74d6083acdf` |
| 35666177 | `942e322083e76f58edea083cde2d66b219b05f9b1ba36d06144f3dcebbec5651` |
| 35666946 | `6e81e45b648b0cfcc1b81fed9b8d0a37096c1c9c4305399aef01178c0f3b794e` |
| 40897069 | `90de156cfec7bcbaf1196b30f042f4c3ecca113e2485084d43b4376b64a1ab4c` |
| 40917149 | `df1912a822a3befb1f74b024f5a30cb1d975b9451155309e9f27be2a2b562d80` |
| 42363405 | `02dd2fb3d42cd65a991ed361a2ab95a1ed99109a9e4f249c1abb1f54d160abc8` |
| 42363636 | `a3e65510dae45692b04c4959aff6962796cf335536d149c1df01e4cf423c5fe1` |
| 51081754 | `1f180df2e15afa67f95c41d85532ce05f2cb9d34132214781ae1738089d10541` |
| 51102564 | `77687e21e217051037664a5d6d30fee786986e75a75fb46b9f6403dbd3835d53` |
| 54220859 | `ebc8d99276e0158292b69f6728cfad065986a84f29c5eb05dddee8587a229bcc` |
| 54220863 | `d27bfdc05bd135f086b68a631f146ece361d3e967e4773d8c6b156d57e4b1022` |

Each of the ten pairs is ONE passage under the declared chunk policy (a longer post is cut in more,
and the cut then depends on the embed-time tokenizer; see the family's docstring). The point id of
each row equals the id the live pack holds for the same question: `tests/test_aw_family_stackoverflow_qa.py`
pins them.

## How the cut was made

`tests/fixture_tools/cut_stackoverflow_qa.py stackoverflow.com-Posts.7z OUT_DIR` streams `Posts.xml` out
of the archive (needs the `7z` program), copies the 24 rows named in it and writes the file above. It
is deterministic: the same archive gives the same file. It was run once, on the operator's copy of the
archive, and its output is this file. The ten questions were picked from the live tranches for being
short, for covering the three licence versions and six question/answer licence pairs, and for being
one passage each. A test (`tests/test_aw_family_stackoverflow_qa.py`) checks the file's sha256, every
row's sha256 and the cut tool against this note.

## Personal data

The ten pairs were picked so that no row names or numbers a person as its owner: every question and
answer row carries `OwnerDisplayName="userNNN"`, the name the dump gives to a deleted account, and no
`OwnerUserId`. A few rows carry `LastEditorUserId` (a number, no name) or `LastEditorDisplayName="userNNN"`.
The four negative rows carry an `OwnerUserId` (a number) and no name of any kind. `Users.xml` (profiles:
names, locations, "about me" text) is NOT in the sample. The texts of the posts are unedited. They were read
for the names of persons, handles and e-mail addresses, and the ten pairs and the four negative rows hold
none (two candidate pairs were replaced after that reading: one named the author of a library, one thanked
a user by a Twitter handle). A test checks that no e-mail address and no name other than a deleted
account's is in any attribute.

## Licence and attribution

The content is licensed under Creative Commons Attribution-ShareAlike (CC BY-SA); the version is the
one the dump gives each post in `ContentLicense` (2.5, 3.0 or 4.0;
<https://creativecommons.org/licenses/by-sa/4.0/>). Redistributing the rows requires attribution and
the same licence. Attribution for these files:

> Source: Stack Exchange Data Dump, `stackoverflow.com-Posts.7z` (Stack Overflow), snapshot of
> 2024-04-07, posts 1130479, 1136143, 3283306, 3283326, 9560723, 9815605, 10269701, 10269828, 30676720,
> 30803311, 35666177, 35666946, 40897069, 40917149, 42363405, 42363636, 51081754, 51102564, 54220859
> and 54220863 (and, as rows the selection refuses, 1311, 10595, 38784 and 38834), by their authors as
> named in each row (deleted accounts at the time of the dump), licensed under CC BY-SA 2.5, 3.0 or 4.0
> as each row's `ContentLicense` says. The page at `https://stackoverflow.com/questions/<id>` is the
> canonical page of each question.

Each index row the family makes carries the licence name (`CC-BY-SA-2.5`, `-3.0`, `-4.0`) and the
canonical question URL. The authors' display names, the profile URLs (when a row has an owner id), the
post ids, the scores and the licence versions are kept in the master-only part of the row and are
removed from the redistributable dump.
