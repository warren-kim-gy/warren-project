WITH tmp_table_01 AS (
  SELECT column_a AS user_id,
    column_b AS platform,
    DATE_ADD(column_c, INTERVAL 9 HOUR) AS kr_date_id,
    CASE
      WHEN column_d = 'Subscription Renewal' THEN 'y'
      ELSE 'n'
    END AS is_sub_renewal,
    SUM(column_e) AS total_price
  FROM schema_name.table_x
  WHERE column_f = '1'
  GROUP BY column_a,
    column_b,
    kr_date_id
),
tmp_table_02 AS (
  SELECT column_g AS id,
    column_h AS country,
    JSON_UNQUOTE(JSON_EXTRACT(column_i, '$.Ad_Group')) AS ad_group,
    JSON_UNQUOTE(JSON_EXTRACT(column_i, '$.Ad_Creative')) AS ad_creative,
    JSON_UNQUOTE(JSON_EXTRACT(column_i, '$.Sub_Publisher')) AS sub_publisher,
    DATE(DATE_ADD(column_c, INTERVAL 9 HOUR)) AS kr_signup_date_id
  FROM schema_name.table_y
),
tmp_final_table AS (
  SELECT DATE_FORMAT(t1.kr_date_id, '%Y-%m-%d %H:%i:%s') AS kr_date_id,
    t1.user_id,
    t1.platform,
    t2.country,
    t1.is_sub_renewal,
    t1.total_price,
    t2.ad_group,
    t2.ad_creative,
    t2.sub_publisher,
    t2.kr_signup_date_id
  FROM tmp_table_01 t1
    LEFT OUTER JOIN tmp_table_02 t2 ON t1.user_id = t2.id
)
SELECT kr_date_id,
  user_id,
  platform,
  country,
  is_sub_renewal,
  ROUND(SUM(total_price), 2) AS total_price,
  CASE
    WHEN ad_group LIKE '%mai-%' THEN 'mai'
    WHEN ad_group LIKE '%aeo-%' THEN 'aeo'
    WHEN ad_group LIKE '%linkclick-%' THEN 'linkclick'
    ELSE NULL
  END AS adset_goal,
  CASE
    WHEN ad_group LIKE '%-aos%' THEN 'aos'
    WHEN ad_group LIKE '%-ios%' THEN 'ios'
    WHEN ad_group LIKE '%-web%' THEN 'web'
    ELSE NULL
  END AS os_info,
  SUBSTRING_INDEX(SUBSTRING_INDEX(ad_group, '_', -2), '_', 1) AS target_info,
  SUBSTRING_INDEX(ad_group, '_', -1) AS campaign_start_date,
  ad_creative,
  sub_publisher,
  kr_signup_date_id
FROM tmp_final_table
WHERE country IS NOT NULL
  AND country NOT IN ('KR', 'PH')
  AND kr_date_id BETWEEN '2024-12-30 00:00:00' AND '2024-12-30 23:59:59'
GROUP BY kr_date_id,
  platform,
  country,
  is_sub_renewal,
  ad_group,
  ad_creative,
  sub_publisher
ORDER BY kr_date_id,
  country,
  is_sub_renewal,
  platform,
  ad_group,
  ad_creative,
  sub_publisher;
