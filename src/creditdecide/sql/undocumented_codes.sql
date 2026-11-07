-- Values of {column} outside the documented code list ($codes), with their counts.
-- ANTI JOIN keeps the rows of data that have no match in the code list.
SELECT {column}            AS code,
       COUNT(*)::BIGINT    AS n
FROM data
ANTI JOIN (SELECT UNNEST($codes) AS code) AS documented
       ON data.{column} = documented.code
WHERE {column} IS NOT NULL
GROUP BY ALL
ORDER BY ALL
