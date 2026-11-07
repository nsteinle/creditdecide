-- Rows per label value, in label order.
SELECT {target} AS label, COUNT(*)::BIGINT AS n
FROM data
GROUP BY ALL
ORDER BY ALL
