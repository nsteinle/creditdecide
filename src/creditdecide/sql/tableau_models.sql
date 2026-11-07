-- Test-split metrics in long form (one row per model and metric), which Tableau filters and
-- colours more easily than one column per metric.
UNPIVOT metrics
ON COLUMNS(* EXCLUDE (model))
INTO NAME metric VALUE value
ORDER BY ALL
