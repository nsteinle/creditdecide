-- How often each factor is a stated reason for flagged test accounts, by rank.
SELECT factor,
       rank,
       COUNT(*)::BIGINT AS accounts,
       AVG(impact)      AS mean_impact
FROM reasons_detail
WHERE is_reason
GROUP BY ALL
ORDER BY rank, accounts DESC, factor
