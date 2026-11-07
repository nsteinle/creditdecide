-- The matched rows of the outcomes join, for the ranking and calibration metrics.
SELECT s.{id} AS id, s.score, s.probability, s.flag::INTEGER AS flag,
       (CAST(l.{target} AS VARCHAR) = $positive_label)::INTEGER AS y
FROM scores AS s
JOIN labels AS l ON s.{id} = l.{id}
ORDER BY s.row_id
