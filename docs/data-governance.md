# Data provenance and publication boundary

The official MovieLens 100K stable release is dated April 1998. The archive contains 100,000 ratings, 943 users, and 1,682 movies. It was retrieved from `https://files.grouplens.org/datasets/movielens/ml-100k.zip` on 2026-09-23. Provenance and actual checksum are in `reports/full/metrics.json`.

The [publisher's terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) allow research subject to attribution and non-endorsement conditions. Redistribution requires separate permission, as does commercial or revenue-bearing use. A portfolio demonstrates independent research; it is not permission to run a commercial recommender. Publication of trained weights is not expressly resolved by these terms, so this project does not assume that compact weights evade the restriction.

## Schema and acquisition

`u.data`: tab-separated integer user ID, movie ID, rating 1–5, Unix UTC timestamp. The code validates the exact row count, rating range, and unique user/movie pairs. `u.item` is actually pipe-delimited, despite inconsistent delimiter wording in the publisher README; the parser validates 24 fields and 1,682 movies. The model uses only IDs, ratings, and timestamps. Titles are read only for the private local demo export. No demographic files are extracted or used.

The publisher MD5 (`0e33842e24a9c977be4e0107933c0723`) verifies the archive identity against its [checksum file](https://files.grouplens.org/datasets/movielens/ml-100k.zip.md5). MD5 is not a modern adversarial integrity guarantee; HTTPS verification is retained and the retrieved file's SHA-256 is recorded (`50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`). Only named archive members are extracted, avoiding arbitrary ZIP paths.

## Publication allowlist

Public: original source, docs, configs, aggregate metrics, benchmark selection logs, screenshots showing only fictional titles, and an original fictional sandbox generated independently of MovieLens.

Private/ignored: source archive, raw ratings, movie metadata, user histories, per-user evaluation arrays, checkpoints, item similarities, and MovieLens browser bundles. `scripts/build_site.py` copies a fixed list of site files and aggregate metrics; it never copies the repository recursively. The bundle has `restricted: true`; the public sandbox has `restricted: false`. The build checks the sandbox designation.

Importing a private bundle is a file read inside the browser, not a server upload. The app uses `File.text()`, validates dimensions and finite numbers, writes titles with `textContent`, and keeps values in memory without localStorage, cookies, telemetry, or external resources. Reloading clears the imported bundle. Do not attach private model bundles to GitHub issues or publish them as build artifacts.

## Interpretation

Missing ratings combine unknown exposure and unknown preference. Sampled negatives are an optimization convention, not observed dislikes. Popularity counts reflect both exposure and taste. The cohort contains users with at least 20 ratings in the full historical dataset; that is source curation, not a training-time eligibility rule. The published pipeline does not use full-period activity to select or map users.
