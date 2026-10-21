# Control Route

Purpose: maintain the shared scientific vocabulary, route registry, immutable
inputs, and transition proposals. This route does not launch simulations or
change accepted counts.

Responsibilities:

- keep 1-D and 2-D evidence separate;
- prevent duplicate fingerprints and output collisions;
- register prospective policy before results exist;
- reconcile route summaries with the canonical controller;
- require independent reproduction before acceptance.

The control route also maintains the shared decision and stop-rule register,
records which items are canonical, proposed, or decision-required, and checks
that every future matrix proposal names its model family, evidence namespace,
output root, fingerprint, preflight, and stop conditions.

It may update this organizational layer without a canonical transaction only
when the change grants no launch authority, changes no accepted count, and
does not alter canonical files. Any executable transition must follow the full
transaction in `../../LIA_EXECUTION_CONTRACT.md` and requires explicit user
approval.
