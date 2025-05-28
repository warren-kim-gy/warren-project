-- ga4 회원가입(signup_complete)이벤트 기준 리텐션 쿼리

WITH tmp_table_01 AS (
  SELECT
    SAFE_CAST(user_id AS INT64) AS user_id,
    DATE(TIMESTAMP_MICROS(event_timestamp), "Asia/Seoul") AS cohort_date
  FROM `warren_sample_project.analytics_000000000.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20250201' AND '20250228'
    AND event_name = 'signup_complete'
    AND user_id IS NOT NULL
    AND geo.country NOT IN ('South Korea', 'Philippines')
), -- cohort_users
tmp_table_02 AS (
  SELECT
    SAFE_CAST(user_id AS INT64) AS user_id,
    DATE(TIMESTAMP_MICROS(event_timestamp), "Asia/Seoul") AS event_date
  FROM `warren_sample_project.analytics_000000000.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20250201' AND '20250331'
    AND user_id IS NOT NULL
    AND geo.country NOT IN ('South Korea', 'Philippines')
), -- event_logs
tmp_final_table AS (
  SELECT
    t1.user_id,
    t1.cohort_date,
    t2.event_date
  FROM tmp_table_01 t1
  LEFT JOIN tmp_table_02 t2
    ON t1.user_id = t2.user_id
) -- joined
SELECT
  cohort_date,
  user_id,
  MAX(IF(event_date = cohort_date + 1, 1, 0)) AS d1,
  MAX(IF(event_date = cohort_date + 2, 1, 0)) AS d2,
  MAX(IF(event_date = cohort_date + 3, 1, 0)) AS d3,
  MAX(IF(event_date = cohort_date + 4, 1, 0)) AS d4,
  MAX(IF(event_date = cohort_date + 5, 1, 0)) AS d5,
  MAX(IF(event_date = cohort_date + 6, 1, 0)) AS d6,
  MAX(IF(event_date = cohort_date + 7, 1, 0)) AS d7,
  MAX(IF(event_date = cohort_date + 8, 1, 0)) AS d8,
  MAX(IF(event_date = cohort_date + 9, 1, 0)) AS d9,
  MAX(IF(event_date = cohort_date + 10, 1, 0)) AS d10,
  MAX(IF(event_date = cohort_date + 11, 1, 0)) AS d11,
  MAX(IF(event_date = cohort_date + 12, 1, 0)) AS d12,
  MAX(IF(event_date = cohort_date + 13, 1, 0)) AS d13,
  MAX(IF(event_date = cohort_date + 14, 1, 0)) AS d14,
  MAX(IF(event_date = cohort_date + 15, 1, 0)) AS d15,
  MAX(IF(event_date = cohort_date + 16, 1, 0)) AS d16,
  MAX(IF(event_date = cohort_date + 17, 1, 0)) AS d17,
  MAX(IF(event_date = cohort_date + 18, 1, 0)) AS d18,
  MAX(IF(event_date = cohort_date + 19, 1, 0)) AS d19,
  MAX(IF(event_date = cohort_date + 20, 1, 0)) AS d20,
  MAX(IF(event_date = cohort_date + 21, 1, 0)) AS d21,
  MAX(IF(event_date = cohort_date + 22, 1, 0)) AS d22,
  MAX(IF(event_date = cohort_date + 23, 1, 0)) AS d23,
  MAX(IF(event_date = cohort_date + 24, 1, 0)) AS d24,
  MAX(IF(event_date = cohort_date + 25, 1, 0)) AS d25,
  MAX(IF(event_date = cohort_date + 26, 1, 0)) AS d26,
  MAX(IF(event_date = cohort_date + 27, 1, 0)) AS d27,
  MAX(IF(event_date = cohort_date + 28, 1, 0)) AS d28
FROM tmp_final_table
GROUP BY cohort_date, user_id
ORDER BY cohort_date, user_id;
