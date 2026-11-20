# Sample: Stack Exchange Q&A (`stackexchange_nonso_qa_p0`)

Six questions and seven of their answers from ONE small Stack Exchange site, Information Security
(`security.stackexchange.com`): 13 `<row>` lines of its `Posts.xml`, whole and byte for byte as the
dump ships them. 14,131 bytes in all. The family reads this directory with
`cyt source run stackexchange_nonso_qa_p0 --from data/samples/stackexchange-nonso --out STAGE`
(one sub-directory per site, named by the site's host, holding that site's `Posts.xml`).

## Origin

- Publisher: Stack Exchange, Inc. Dataset: "Stack Exchange Data Dump", the public archive of the
  network's user-contributed content, kept at <https://archive.org/details/stackexchange>.
- File: `security.stackexchange.com.7z`, 260,037,012 bytes, snapshot of 2024-04-07 (the dates inside
  the archive). Its sha256 is `cbd0cd8eeb565913c37328af7c0a729e77f48d283d9f917761ad7e76c84c3b62`
  (the archive's own listing gives md5 `cc6de9272d9a9edcecdbbfd4f85a8b9e`, sha1
  `1f1ec62c54e478061be2a35b55c7be4f397e4f75`). The archive itself is not in this repository.

## The file

`security.stackexchange.com/Posts.xml`: 14,131 bytes, sha256
`70ca281d449331fac78e7cbfd28d354f88756f49532740b90eed8e26930f2565`. It has the dump file's own first
two lines (byte-order mark and XML declaration, `<posts>`), the 13 rows below in the dump's order,
and the closing `</posts>`. No row is edited, shortened or re-encoded; no attribute is removed.

| question | its answers in the sample (score) | licence per the answer's date (what the extractor stamps) |
|---|---|---|
| 16941 | 16942 (3), 17138 (4): two "Top answer" passages | CC-BY-SA-3.0 |
| 24832 | 24833 (5), accepted | CC-BY-SA-3.0 |
| 63305 | 63309 (4), accepted | CC-BY-SA-3.0 |
| 143964 | 144186 (10), accepted | CC-BY-SA-3.0 by date; the row's own `ContentLicense` says 4.0 |
| 189582 | 189589 (4) | CC-BY-SA-4.0 |
| 266446 | 266456 (6) | CC-BY-SA-4.0 |

| row (post id) | sha256 of the row line |
|---|---|
| 16941 | `c9fbff6685ca0a2f20246e22467e66f95d2e50fd93438e01b7d8b370ef2461a1` |
| 16942 | `4ddc9651cb86d1ea37711f65ad7fbf1f9e59d87750118c32d8d1fec809478e7e` |
| 17138 | `37e5cb5522b678c109f86c67a729a4b04d7e5bd9590dfffda1f12a01d8e79e4c` |
| 24832 | `d10fc11fab75fc25c56a5b7c78b4fc54fb0fdeee1652dfe6582aba92c974fee7` |
| 24833 | `8afa6b9bb57671121c29d31ba390c5d6c9382b11f02c3d59685d2a25d251460f` |
| 63305 | `c28a34210d3376d222a23999a1a14782f3f8c30f3317859bd6fce2cf63780dd6` |
| 63309 | `f4cc6c46960d003f414d32f12fe7b1182b46628f2a7ff447d7fce5027d65f198` |
| 143964 | `dd7b46abce2ed5005e844b7d1ab65b80b512dcf4b6602fbd566dfe2fc8939b95` |
| 144186 | `bdb1c03d5c132b8af4589f001049545e5eae3289327b5e084b988a4370da5fd7` |
| 189582 | `6a5624999073a33826ed7497c1f083b2998551dae80c4edd62700bf8358400ec` |
| 189589 | `7b1c4c71ecd3ec5f5f3b5e15a106660d48e3f2ee4efcdd5e9bc3ebe643809e97` |
| 266446 | `da5e35d321c1c1af84f4259e2f7987225dcb157fc2b7557dbdf62fcbec6b01d8` |
| 266456 | `ac147013589f4843456c253affac9c99c024855d07d15c1df12a39be561f3616` |

Only answers that score 3 or more are in the sample (the extractor ignores the others); each
question's other answers are not. So the sample is a selection of rows, not a whole thread.

## How the cut was made

`tests/fixture_tools/cut_stackexchange_nonso.py ARCHIVE.7z OUT_DIR` streams `Posts.xml` out of the
archive (needs the `7z` program) and copies the 13 rows named in `POST_IDS`. It is deterministic:
the same archive gives the same file. The six questions were picked for being short and for covering
two licence eras (3.0 and 4.0) and two answer shapes (accepted, and top-scored where the question has
no accepted answer in the sample). A
test (`tests/test_aw_family_stackexchange_nonso_qa.py`) checks the file's sha256 and every row's
sha256 against this record.

## Personal data

The posts were picked so that no row names or numbers a person as its owner: every question and
answer row carries `OwnerDisplayName="userNNN"`, the name the dump gives to a deleted account, and no
`OwnerUserId`. `Users.xml` (profiles: names, locations, "about me" text) is NOT in the sample.
What remains, unmodified, because the rows are whole:

- `LastEditorUserId` (a number, no name) on three rows, and `LastEditorDisplayName="userNNN"` on
  four: the editors of those posts. No name of a natural person appears in any attribute.
- The text of the posts. A post body can name people or sites; these six hold public links
  (Wikipedia, OWASP, NIST, OpenSSL) and no e-mail address.

## Licence and attribution

The content is licensed under Creative Commons Attribution-ShareAlike (CC BY-SA); the version is the
one in force when the post was made: 2.5 up to 2011-04-08, 3.0 up to 2018-05-02, 4.0 after
(<https://creativecommons.org/licenses/by-sa/4.0/>; the dump's rows say it in `ContentLicense`).
Redistributing the rows requires attribution and the same licence. Attribution for these files:

> Source: Stack Exchange Data Dump, `security.stackexchange.com` (Information Security Stack
> Exchange), snapshot of 2024-04-07, posts 16941, 16942, 17138, 24832, 24833, 63305, 63309, 143964,
> 144186, 189582, 189589, 266446 and 266456, by their authors as named in each row (accounts deleted
> at the time of the dump), licensed under CC BY-SA 3.0 or 4.0 as each row's `ContentLicense` says.
> The post at `https://security.stackexchange.com/a/<answer id>` is the canonical page of each answer.

Each index row the family makes carries the licence name (`CC-BY-SA-3.0`, `CC-BY-SA-4.0`) and the
canonical answer URL. The author's display name, the profile URL (when the row has an owner id), the
post ids, the score and the licence version are kept in the master-only part of the row and are
removed from the redistributable dump.
