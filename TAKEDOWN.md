# Content removal / opt-out (take-down) — websear.ch (Cytir)

Cytir is a finder of authoritative URLs, not a content host: the index stores
only minimal records (URL, title, short snippet, embedding, sparse terms) and
links back to your live pages. If you do not want your site referenced,
we will remove it.

## Contact

**bender@impactia.org** — subject line: `TAKEDOWN <your-domain>`

Please include: the (sub)domain or URLs to remove, and (for verification)
either a message from an email address at that domain or a note placed at a
URL on the site. No legal formalities are required; a plain request is enough.

## What we do on request (procedure)

1. **Acknowledge within 3 business days.**
2. **Verify** the requester controls the domain (domain email or a token page).
3. **Delete** all passages for the (sub)domain from the serving index
   (Qdrant delete-by-filter on the indexed `domain` payload field) and from
   staged/raw pipeline files; confirm deletion by count.
4. **Blocklist** the (sub)domain in the ingest allowlist/queue so future
   builds and refresh passes skip it (consent cache entry pinned declined).
5. **Offer the card fallback**: unless you object, we may keep a minimal
   *source card* (domain name, topics it is authoritative for, a link to your
   own search page — no page content). Say "no card" and we drop that too.
6. **Confirm completion** by reply, normally within 7 days of verification.

## Register-published individuals — person-name fast path

Some commercial-register entity cards (Zefix/SOGC-sourced) carry the
register-published signatories/board/management for that company, per
`sources/POLICY.md@v3.1 §4.9.1` (board-ratified, "register-published
individuals"). If your name appears this way and you want it removed, this
route is open to **anyone, anywhere** — we honor it the same way regardless
of your jurisdiction or nationality; it is not gated on GDPR/LPD applying
to you specifically (see POLICY §4.9.1 "Values and lawful basis").

1. Email **bender@impactia.org**, subject `TAKEDOWN PERSON <your name>`.
   Include the company name or UID your name is associated with — no domain
   email or token page is required (you don't control a domain here); the
   company/UID + name is enough to identify what to remove.
2. **Acknowledge within 3 business days, resolve within 7** (same SLA as a
   domain takedown).
3. We remove your name/token from that card's search text; **the company
   card itself is not deleted** (it remains independently admissible on its
   own identity/purpose). The removal is pinned so a later register-refresh
   pass does not silently re-add it without a fresh review.
4. If the register itself later shows you as departed from the role, we
   remove the token automatically at the next register pass — no request
   needed.

## Standing opt-out signals (no email needed)

Cytir re-resolves consent live per source (never inherited from Common Crawl)
and honours, in precedence order: AIPREF `Content-Usage`, `robots.txt`
(including training-only tokens such as `*-Extended`; blocking `CCBot`
also removes you, since our copies come via Common Crawl), and W3C TDMRep
(`tdm-reservation: 1` in `/.well-known/tdmrep.json` or robots.txt). Signals
are re-checked at build/refresh time (≤30-day cache), so publishing an
opt-out removes your content from subsequent builds automatically — email
us if you want it out faster.
