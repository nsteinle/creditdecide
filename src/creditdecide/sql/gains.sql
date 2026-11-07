-- Cumulative gains and lift by score decile (decile 1 = highest scores).
-- Rows are ranked by score, ties broken by input order, and cut into $bins groups whose sizes
-- differ by at most one (the first $n % $bins groups get the extra row), as numpy's
-- array_split does.
WITH ranked AS (
    SELECT y, score,
           ROW_NUMBER() OVER (ORDER BY score DESC, row_id) - 1 AS pos
    FROM scored
),
sizes AS (
    SELECT $n // $bins AS q, $n % $bins AS r
),
binned AS (
    SELECT ranked.*,
           1 + CASE WHEN pos < r * (q + 1) THEN pos // (q + 1)
                    ELSE r + (pos - r * (q + 1)) // q END AS decile
    FROM ranked, sizes
),
per_decile AS (
    SELECT decile,
           COUNT(*)::BIGINT   AS accounts,
           MIN(score)         AS min_score,
           SUM(y)::BIGINT     AS positives
    FROM binned
    GROUP BY ALL
)
SELECT decile::BIGINT                                         AS decile,
       accounts,
       min_score,
       positives,
       positives / accounts                                   AS positive_rate,
       SUM(positives) OVER cumulative / NULLIF($positives, 0) AS cumulative_capture,
       (SUM(positives) OVER cumulative / SUM(accounts) OVER cumulative)
           / NULLIF($positives / $n, 0)                       AS cumulative_lift
FROM per_decile
WINDOW cumulative AS (ORDER BY decile)
ORDER BY decile
