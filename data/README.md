# Processed data

Place released processed datasets in this directory. Dataset contents are
ignored by Git; this README is the only tracked file.

For a dataset named `<Dataset>`, SAGE expects:

```text
data/<Dataset>/
  <Dataset>.index.json
  <Dataset>.item.json
  <Dataset>.train.inter
  <Dataset>.valid.inter
  <Dataset>.test.inter
```

`index.json` maps item IDs to four RQ-SID tokens and, when necessary, an
additional collision-resolution token:

```json
{
  "item_1": ["<|a_12|>", "<|b_7|>", "<|c_93|>", "<|d_4|>"]
}
```

`item.json` maps item IDs to metadata fields such as `title`, `categories`,
`brand`, and `description`. Each tab-separated `.inter` file has a header and
three columns: user ID, a space-separated interaction history, and the target
item ID.

Validate the files with:

```bash
python scripts/check_data.py --data_dir data/<Dataset> --dataset <Dataset>
```
