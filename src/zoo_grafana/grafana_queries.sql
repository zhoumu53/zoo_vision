################################################################
# Hour counter on top
WITH cte AS (
  SELECT t.identity_id as id
    , date_bin('1 second', o.time, TIMESTAMPTZ '${__from:date}') AS time_bined
    , count(*)>2 as is_sleeping
  FROM observations AS o
  INNER JOIN tracks AS t ON t.id=o.track_id 
  WHERE  $__timeFilter(o.time) 
    AND o.behaviour_id IN (2,3)
  GROUP  BY t.identity_id, time_bined
)
SELECT ti.name
  , DATE_TRUNC('hour', s.time_bined) as display_time_bined
  , sum(cast(coalesce(cte.is_sleeping,false) as int))*1.0 / (60.0*60) as hours_sleeping
FROM (
      SELECT generate_series(TIMESTAMPTZ '${__from:date}', TIMESTAMPTZ '${__to:date}', interval '1 second')
    ) s(time_bined)
    CROSS JOIN (SELECT id,name FROM identities) ti
    LEFT OUTER JOIN cte
      USING (time_bined, id)
GROUP BY ti.name, display_time_bined
ORDER BY min(ti.id), display_time_bined

################################################################
# Observability
WITH cte AS (
   SELECT i.id
    , date_bin('10 second', o.time, TIMESTAMPTZ '${__from:date:YYYY-MM-DD}') AS time_bined
    , count(*) as observation_count
   FROM observations AS o
   INNER JOIN tracks AS t ON t.id=o.track_id 
   INNER JOIN identities AS i on t.identity_id=i.id
   WHERE  $__timeFilter(o.time) 
   GROUP  BY i.id, time_bined
)
SELECT ti.name
   , s.time_bined
   , cast(coalesce(cte.observation_count,0)>0 as int)*0.5+ti.id as observed
   , ti.color as color
   -- , '#777777' as color
FROM  (
   SELECT generate_series(TIMESTAMPTZ '${__from:date:YYYY-MM-DD}', TIMESTAMPTZ '${__to:date}', interval '10 second')
   ) s(time_bined)
   CROSS JOIN identities as ti
   LEFT JOIN cte USING (time_bined, id)
WHERE ti.id != 0
ORDER  BY s.time_bined;

################################################################
# Laterality
WITH cte AS (
  SELECT i.id
    , date_bin('10 second', o.time, TIMESTAMPTZ '${__from:date:YYYY-MM-DD}') AS time_bined
    , count(*) FILTER (WHERE behaviour_id=2) as sleeping_left_count
    , count(*) FILTER (WHERE behaviour_id=3) as sleeping_right_count
  FROM observations AS o
  INNER JOIN tracks AS t ON t.id=o.track_id 
  INNER JOIN identities AS i on t.identity_id=i.id
  WHERE  $__timeFilter(o.time) 
  GROUP  BY i.id, time_bined
)
SELECT ti.name
  , s.time_bined
  , ti.id + 0.3*cast(coalesce(cte.sleeping_left_count,0)>0 as int) as sleeping_left
  , ti.id - 0.3*cast(coalesce(cte.sleeping_right_count,0)>0 as int) as sleeping_right
FROM  (
   SELECT generate_series(TIMESTAMPTZ '${__from:date:YYYY-MM-DD}', TIMESTAMPTZ '${__to:date}', interval '10 second')
   ) s(time_bined)
   CROSS JOIN (SELECT id,name FROM identities WHERE id!=0) ti
   LEFT JOIN cte USING (time_bined, id)
ORDER  BY s.time_bined;
