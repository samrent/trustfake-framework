# Conventions for this brain

Type-flattened: a leaf lives in the folder named for its role, each folder has its own
`MANIFEST.md`, and the root `MANIFEST.md` points at those. A leaf with no MANIFEST line is
invisible to `context-load`; a MANIFEST line with no leaf dangles. Keep both in sync.

Folders: `reference/` (how things work, evergreen) · `decisions/` (a choice + why) ·
`gotchas/` (trap -> fix) · `rejected/` (hypothesis -> what killed it) ·
`investigations/` (dated, append-only) · `snapshots/` (dated numbers) ·
`handoff/` (current state of unfinished work) · `specs/` · `archive/`.

Frontmatter: `type`, `title`, `status` (current|stale|superseded|resolved|active), `as_of`,
`source`, `tags`, `links`.

House rules that matter here: **numbers go in `snapshots/`, mechanism in `reference/`**, so
reference stays evergreen. Absolute dates only. Explain the *why* or it gets re-litigated.
Update in place rather than creating a near-duplicate.
