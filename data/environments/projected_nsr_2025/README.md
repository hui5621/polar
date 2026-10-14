# Fixed projected NSR scenes

The twelve `.npz` files are the fixed daily environment grids used by this code package. See `manifest.json` for dates, map dimensions, projection, start/goal cells and layer names.

Arrays use shape `(rows, cols) = (325, 209)` and 18 km nominal grid spacing. Coordinates use `(row, col)` with row 0 at the top. Only processed environment layers are present; original observations, chart files, route traces and local source paths are omitted. Data attribution is in the repository's [`DATA_SOURCES.md`](../../../DATA_SOURCES.md).
