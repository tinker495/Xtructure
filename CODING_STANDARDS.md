# Coding Standards & Code Review Rules

Write repository documentation in English and review findings in Korean.
Explain non-obvious invariants in comments. Keep experiment records in local `runs/`.

## Structural review

Look beyond the edited lines for duplicated ownership, shotgun surgery, delegation-only wrappers and eroded stage boundaries. Prefer deep modules with narrow interfaces. Flag single-implementation interfaces, unused configuration and replacements for standard-library functionality when deletion would simplify the design. Present findings in severity order with evidence and consequences. Review new commits before sign-off. Treat private or unsanctioned cross-package imports as boundary violations. CI handles formatting and linting. Test changes require the authorization specified in [AGENTS.md](AGENTS.md).

## Cross-repository ownership

Trace each changed boundary operation from its caller to the documented public API, including helpers assembled entirely from public calls. Ask which repository must change if the provider changes its representation while preserving the public contract. A caller that must change with the representation has taken ownership of a provider invariant.

Xtructure owns container layout, hash probing, padding and duplicate-write resolution. For conditional writes, callers supply row indices, values and an eligibility mask. Keep slot sentinels and first-writer election inside the update operation. Check affected public callers when changing it. Puzzle legality belongs to PuXle and search priorities belong to JAxtar.

In the review or completion summary, name the caller, the public API and the repository responsible for the invariant. For a violation, cite the offending call and the operation the owner should provide. Require that correction before approval, even when the import guard passes or a benchmark improves. For a diff with no affected boundary, state that once. Recheck changed calls on subsequent revisions.
