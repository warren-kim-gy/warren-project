WITH tmp_table_01 AS (
  SELECT
    INT64(column_a) AS fb_campaign_id,
    MAX(STRING(column_b)) AS fb_campaign_name
  FROM
    `schema_name.table_x`
  WHERE
    column_c = '2023-05-18'
    AND JSON_VALUE(column_d) = '12345678'
  GROUP BY fb_campaign_id
)
SELECT
  column_e AS country,
  tmp_table_01.fb_campaign_name,
  COUNT(DISTINCT(CASE WHEN column_f = 1 THEN column_g ELSE NULL END)) AS NRU_cnt,
  COUNT(DISTINCT(CASE WHEN column_f > 1 THEN column_g ELSE NULL END)) AS RU_cnt
FROM
  `schema_name.table_y` tmp_table_02
LEFT OUTER JOIN
  tmp_table_01 ON
  SAFE_CAST(JSON_VALUE(tmp_table_02.column_h, '$.hidden_column_name') AS INT64) = tmp_table_01.fb_campaign_id
WHERE
  column_i = '2023-05-18'
  AND column_j = 'sample_event_complete'
  AND JSON_VALUE(column_h, '$.entry') = 'ad'
GROUP BY country, fb_campaign_name;
