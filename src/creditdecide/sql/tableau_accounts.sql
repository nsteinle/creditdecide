-- One row per test-split account: its inputs, readable group names, the decision model's
-- score and calibrated probability, the decision, the score decile (the same deciles as
-- gains_test.csv) and, for flagged accounts, the top reason code.
WITH test AS (
    SELECT *, ROW_NUMBER() OVER (ORDER BY decision_score DESC, row_id) - 1 AS pos
    FROM samples
    WHERE split = 'test'
),
sizes AS (
    SELECT COUNT(*) // 10 AS q, COUNT(*) % 10 AS r FROM test
)
SELECT test.* EXCLUDE (split, pos, row_id, decision_score, probability, flag, {target}),
       test.{target}                                      AS actual,
       test.decision_score                                AS score,
       test.probability,
       test.flag,
       CASE WHEN test.flag = 1 THEN 'flagged' ELSE 'not flagged' END AS decision,
       1 + CASE WHEN pos < r * (q + 1) THEN pos // (q + 1)
                ELSE r + (pos - r * (q + 1)) // q END   AS score_decile,
       reasons.reason_1                                   AS top_reason,
       reasons.impact_1                                   AS top_reason_impact
FROM test
CROSS JOIN sizes
LEFT JOIN reasons ON reasons.row = test.row_id
ORDER BY score DESC, test.row_id
