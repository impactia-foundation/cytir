# Federal Register govinfo fixtures

Five real issues, fetched 2026-10-03 (read-only, one request each, honest user agent) from
`https://www.govinfo.gov/bulkdata/FR/{year}/{mm}/FR-{date}.xml`: 2000-01-18, 2005-06-15, 2010-03-01,
2017-08-02, 2025-09-30. `fx-FR-*.xml` keeps 3-4 whole documents per issue (one rule, one proposed
rule, two notices where the issue has short ones) under the original `<FEDREG>` header; nothing in a
kept document is edited.

`fx-api-*.json`: the federalregister.gov documents API listing for the same publication day
(`document_number`, `title`, `abstract`, `type`, `publication_date`, `raw_text_url`, `html_url`,
`citation`, `agencies`), reduced to the documents in the XML plus one listed document that is not in
the trimmed XML (`in_xml` names the document numbers that are). Public domain (17 U.S.C. 105).
