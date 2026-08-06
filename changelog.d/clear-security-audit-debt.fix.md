Clear the security audit on `main`: bump `cryptography` 48.0.1 → 50.0.0
(GHSA-g6cj-pr64-35w5, GHSA-m2h6-j472-rp4c, GHSA-jwv3-5hgf-82ww and their PYSEC
equivalents — six advisories) and `pymdown-extensions` 10.21.3 → 11.0.1
(GHSA-9xwg-3r6f-jcx2, a path traversal in the b64 extension). Both are
transitive and were failing the audit on every PR, not just new ones.
