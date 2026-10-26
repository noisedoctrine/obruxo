# Fresh basic confirmation data is reserved, without scoring

The [reservation](data/basic_confirmation_reservation.json) fixes **192 new authored validation roots** and **24 fresh balanced performances** under the [predeclared card](BASIC_CONFIRMATION_RESERVATION_CARD.md). Preparation takes **84.93 seconds / 0.11 GiB RAM**, within ten minutes / 4 GiB, with zero renders, feature extractions, optimizer updates or candidate scores. Five local tests, Ruff and diff checks pass. Implementation `b5296cc` precedes execution.

The deterministic expansion contains 4,096 states: 3,259 train, 400 validation and 437 test. All 512 original states, targets, lineages and split assignments are preserved, including all 64 original test reservations. Only 192 newly added validation roots are selected; the new training states are unused. Each unison/filter combination contributes 32 roots and receives each performance set four times.

The [independent check](data/basic_confirmation_reservation_independent_check.json) verifies the pool/reservation identities, original prefix and splits. None of the selected state or lineage identities overlaps any of **26 prior experimental pool manifests** in the training output directory. The report explicitly excludes unit-test fixture directories; it is not a scan of unrelated storage. All 42 source files pinned by the completed sound fit remain unchanged.

The [measured summary](data/basic_confirmation_reservation_summary.json) and reservation remain local alongside their generated pool. Reservation ID: `ad79b987d567ee7d9e05f20c0393a91d34cbfd5b317955998e5288aebe5aa019`. The existing 512-state source file is unchanged. The card's command reconstructs the extended pool; no multi-megabyte generated pool or audio is committed.

The [sound fit fails its quality gate](TRAINABLE_BASIC_FIT_RESULTS.md), so **no confirmation audio is collected**. These cases remain reserved while A2 continues. A separate complete scoring/recovery plan and a qualifying candidate are required before evaluation; second-seed training, listening and release remain open.
