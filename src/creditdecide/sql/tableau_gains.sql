-- Gains by decile, with the share of accounts reviewed so far: plotting
-- cumulative_capture against share_reviewed next to the diagonal (random selection) gives
-- the gains chart.
SELECT *,
       SUM(accounts) OVER (ORDER BY decile) / SUM(accounts) OVER () AS share_reviewed
FROM gains
ORDER BY decile
