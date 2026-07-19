-- Hand-written workload for the fleet-tracking schema in catalog.json.
--
-- This is the lowest-friction workload source: paste the statements your
-- application issues into a file. Because there are no execution counts, an
-- optional `-- calls: N` hint tells the advisor how often each one runs; without
-- it every statement is counted once and ranking becomes structural only.

-- calls: 1243900
-- Live map: everything near the operator's cursor in the last quarter hour.
SELECT p.id, p.vehicle_id, p.recorded_at
FROM vehicle_positions p
WHERE ST_DWithin(p.geom, ST_SetSRID(ST_MakePoint(536120.5, 178233.9), 3857), 250)
  AND p.recorded_at > now() - interval '15 minutes'
ORDER BY p.recorded_at DESC
LIMIT 500;

-- calls: 86400
-- Geofence roll-up, run once a second by the alerting worker.
SELECT g.id, count(*) AS hits
FROM geofences g
JOIN vehicle_positions p ON ST_Contains(g.geom, p.geom)
WHERE g.active = true
GROUP BY g.id;

-- calls: 241800
-- Dispatch board: active trips crossing the viewport for one fleet.
SELECT t.id, t.started_at, t.fleet_id
FROM trips t
WHERE ST_Intersects(t.geom, ST_MakeEnvelope(535000, 175000, 541000, 181000, 3857))
  AND t.fleet_id = 12
  AND t.status = 'active';

-- calls: 412700
-- Nearest available drivers to a pickup point.
SELECT d.id, d.driver_id
FROM driver_pings d
ORDER BY d.geom <-> ST_SetSRID(ST_MakePoint(536120.5, 178233.9), 3857)
LIMIT 5;

-- calls: 51200
-- Zone occupancy report over a large window.
SELECT z.zone_id, count(*)
FROM zone_visits z
WHERE z.geom && ST_MakeEnvelope(520000, 160000, 570000, 195000, 3857)
GROUP BY z.zone_id;

-- calls: 32400
-- Anti-pattern: a distance comparison no index can answer.
SELECT p.id
FROM vehicle_positions p
WHERE ST_Distance(p.geom, ST_SetSRID(ST_MakePoint(536120.5, 178233.9), 3857)) < 400
ORDER BY p.recorded_at DESC
LIMIT 20;

-- calls: 7300
-- Anti-pattern: reprojecting the indexed column on every row.
SELECT t.id
FROM trips t
WHERE ST_Intersects(ST_Transform(t.geom, 4326), ST_SetSRID(ST_MakePoint(-0.1276, 51.5072), 4326));
