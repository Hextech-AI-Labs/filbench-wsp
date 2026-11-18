# Development/Test Split Protocol

The dev/test split must be generated once and committed as the manifest in
`configs/dev_split_manifest.json`. It must not be regenerated implicitly by an
evaluation command.

## Rules

- Use the frozen seed in the manifest.
- Store source example identifiers and indices, not only row positions.
- Keep dev and test identifiers disjoint.
- Run all three evaluation conditions with the same split and settings.
- Treat any change to the split as a new experiment.

The source dataset size and exact stratification policy remain pending the
upstream FilBench inspection.
