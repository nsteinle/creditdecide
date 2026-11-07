-- Population stability index of every model input and of the score, new data vs the
-- reference sample saved with the model (creditdecide.monitor):
--     PSI = sum over bins of (actual - expected) * ln(actual / expected)
-- with both shares floored at $eps so empty bins stay finite.
-- Binned variables: a value's bin is the number of reference edges <= the value (numpy's
-- searchsorted, side="right"). Discrete variables: the bin of a known value, or one extra
-- bin for values never seen in the reference.
WITH numeric_bins AS (
    SELECT v.variable, v.row_id, COUNT(e.edge) AS bin
    FROM values_numeric AS v
    LEFT JOIN edges AS e ON e.variable = v.variable AND e.edge <= v.value
    GROUP BY ALL
),
discrete_bins AS (
    SELECT v.variable, v.row_id, COALESCE(k.bin, u.unseen_bin) AS bin
    FROM values_text AS v
    JOIN unseen AS u USING (variable)
    LEFT JOIN known_values AS k ON k.variable = v.variable AND k.value = v.value
),
counts AS (
    SELECT variable, bin, COUNT(*)::DOUBLE / $n AS actual
    FROM (SELECT * FROM numeric_bins UNION ALL SELECT * FROM discrete_bins)
    GROUP BY ALL
),
shares AS (
    SELECT e.variable, e.position, e.bin,
           GREATEST(e.expected, $eps)                AS expected,
           GREATEST(COALESCE(c.actual, 0.0), $eps)   AS actual
    FROM expected AS e
    LEFT JOIN counts AS c USING (variable, bin)
),
per_variable AS (
    -- summed in bin order, so a rerun gives the same last bit whatever the thread count
    SELECT variable, position,
           SUM((actual - expected) * LN(actual / expected) ORDER BY bin) AS psi
    FROM shares
    GROUP BY ALL
)
SELECT variable, psi,
       CASE WHEN psi < $stable THEN 'stable'
            WHEN psi <= $significant THEN 'moderate shift'
            ELSE 'significant shift' END AS status
FROM per_variable
ORDER BY psi DESC, position
