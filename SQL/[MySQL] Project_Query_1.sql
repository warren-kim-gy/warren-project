WITH tmp_table_01 AS (
  SELECT column_a AS user_id,
    column_b AS platform,
    DATE(DATE_ADD(column_c, INTERVAL 9 HOUR)) AS kr_date_id,
    SUM(column_d) AS total_price
  FROM schema_name.table_x
  WHERE column_e = '1'
  GROUP BY column_a,
    column_b,
    kr_date_id
),
tmp_table_02 AS (
  SELECT column_f AS id,
    column_g AS country
  FROM schema_name.table_y
),
tmp_final_table AS (
  SELECT t1.kr_date_id,
    t1.user_id,
    t1.platform,
    t2.country,
    t1.total_price
  FROM tmp_table_01 t1
    LEFT OUTER JOIN tmp_table_02 t2 ON t1.user_id = t2.id
)
SELECT *
FROM tmp_final_table
WHERE country IS NOT NULL
  AND kr_date_id BETWEEN '2024-11-01' AND '2024-11-03';
