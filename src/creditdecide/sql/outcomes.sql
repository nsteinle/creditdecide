-- Outcomes analysis once labels arrive: join the saved decisions to the labels by {id},
-- then count the confusion matrix. Rows without a match on either side are counted, not
-- silently dropped.
WITH joined AS (
    SELECT s.{id} AS id_scores, l.{id} AS id_labels,
           s.flag::INTEGER                                       AS flag,
           (CAST(l.{target} AS VARCHAR) = $positive_label)::INTEGER AS y
    FROM scores AS s
    FULL OUTER JOIN labels AS l ON s.{id} = l.{id}
)
SELECT COUNT(*) FILTER (WHERE id_scores IS NOT NULL AND id_labels IS NOT NULL)::BIGINT AS n,
       COUNT(*) FILTER (WHERE id_labels IS NULL)::BIGINT     AS scored_without_label,
       COUNT(*) FILTER (WHERE id_scores IS NULL)::BIGINT     AS labelled_without_score,
       COALESCE(SUM(y)    FILTER (WHERE flag IS NOT NULL), 0)::BIGINT AS positives,
       COALESCE(SUM(flag) FILTER (WHERE y IS NOT NULL), 0)::BIGINT    AS flagged,
       COALESCE(SUM(flag * y), 0)::BIGINT                   AS tp,
       COALESCE(SUM(flag * (1 - y)), 0)::BIGINT             AS fp,
       COALESCE(SUM((1 - flag) * y), 0)::BIGINT             AS fn,
       COALESCE(SUM((1 - flag) * (1 - y)), 0)::BIGINT       AS tn
FROM joined
