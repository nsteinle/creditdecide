-- Rows that share every column in {columns} with at least one other row, and how many of
-- those groups disagree on the label. GROUP BY ALL groups by every selected column that is
-- not an aggregate, i.e. by {columns}; NULLs form one group, as in pandas .duplicated().
WITH groups AS (
    SELECT {columns},
           COUNT(*)                  AS n,
           COUNT(DISTINCT {target})  AS labels
    FROM data
    GROUP BY ALL
)
SELECT COALESCE(SUM(n) FILTER (WHERE n > 1), 0)::BIGINT     AS rows_in_shared_groups,
       COUNT(*) FILTER (WHERE n > 1 AND labels > 1)::BIGINT  AS groups_with_conflicting_labels,
       COALESCE(SUM(n - 1), 0)::BIGINT                       AS duplicate_rows
FROM groups
