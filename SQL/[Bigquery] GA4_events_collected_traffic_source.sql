SELECT
  geo.country AS country,
  collected_traffic_source.manual_source AS manual_source,
  collected_traffic_source.manual_medium AS manual_medium,
  collected_traffic_source.manual_campaign_name AS manual_campaign_name,
  collected_traffic_source.manual_content AS manual_content,
  collected_traffic_source.manual_term AS manual_term,
  COUNT(DISTINCT user_pseudo_id) AS unique_total_users,
  COUNTIF(event_name = "click_apply") as click_apply_cnt,
  SUM((SELECT value.int_value FROM unnest(event_params) WHERE KEY = 'engagement_time_msec'))/1000 AS engagement_time_seconds
FROM
  `loki-prod-35e1e.analytics_*********.events_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20240129'  AND '20240129'
  AND geo.country LIKE '%United States%'
  AND collected_traffic_source.manual_source IN ('fb', 'ig')
GROUP BY
  country,
  manual_source,
  manual_medium,
  manual_campaign_name,
  manual_content,
  manual_term
