-- Rebuilds preflight.acled_full and prefligh.acled

DROP TABLE IF EXISTS preflight.acled_full;
DROP TABLE IF EXISTS preflight.acled;

CREATE TABLE preflight.acled_full AS
WITH month_acled AS
         (
             SELECT *,
                    EXTRACT(MONTH FROM event_date :: DATE)               AS month,
                    public.priogrid(latitude::float4, longitude::float4) AS priogrid_gid
             FROM dataprep.acled
             WHERE latitude::float BETWEEN -180 AND 180
               AND longitude::float BETWEEN -90 AND 90
         ),
     month_acled2 AS
         (
             SELECT month_acled.*,
                    staging.month.id AS month_id
             FROM month_acled,
                  staging.month
             WHERE (month_acled.year :: INT = staging.month.year_id AND
                    month_acled.month = staging.month.month)
         )
SELECT *
FROM month_acled2;



ALTER TABLE preflight.acled_full
    ADD COLUMN type_of_violence INT;

ALTER TABLE preflight.acled_full
    ADD COLUMN type_of_protest TEXT;

-- 1. We are emulating UCDP/ViEWS StateBased category using ACLED data.
-- i.e. Military Forces vs. others/other Military Forces, only "battles" and "remote violence"
-- no civilians involved.
-- TODO: shelling and remote violence may need to be treated differently
UPDATE preflight.acled_full
SET type_of_violence = 1
WHERE (event_type ILIKE '%%battle%%' OR event_type ILIKE '%%remote%%')
  AND actor1 || actor2 ILIKE '%%military forces%%'
  AND actor1 || actor2 NOT ILIKE '%%civilians%%';



-- 2. We are emulating UCDP/ViEWS StateBased category using ACLED data.
-- i.e. no military forces, no civilians, only "battles" and "remote violence"
-- UCDP''s artificial organizational criteria are not included and cannot for now be included
UPDATE preflight.acled_full
SET type_of_violence = 2
WHERE (event_type ILIKE '%%battle%%' OR event_type ILIKE '%%remote%%')
  AND actor1 || actor2 NOT ILIKE '%%military forces%%'
  AND actor1 || actor2 NOT ILIKE '%%civilians%%';



-- 3: Emulate UCDP/Views OneSided category.
-- Remote violence, battle and violence against civilians
-- TODO: This may be improved using a better division of "Remote Violence"
UPDATE preflight.acled_full
SET type_of_violence = 3
WHERE (event_type ILIKE '%%battle%%' OR event_type ILIKE '%%remote%%' OR event_type ILIKE '%%civi%%')
  AND actor1 || actor2 ILIKE '%%civilians%%';

-- 4: Protests
-- The entire protest category, as is
UPDATE preflight.acled_full
SET type_of_violence = 4
WHERE event_type ILIKE '%%protest%%';

UPDATE preflight.acled_full
SET type_of_protest = 'p'
WHERE type_of_violence = 4
  AND (inter1::int = 6 OR inter2::int = 6);



UPDATE preflight.acled_full
SET type_of_protest = COALESCE (type_of_protest, '') || 'r'
WHERE
    type_of_violence=4
  AND (inter1::INT =5
   OR inter2::INT =5);



UPDATE preflight.acled_full
SET type_of_protest = COALESCE(type_of_protest, '') || 'x'
WHERE event_type ILIKE '%violence against civi%'
  AND interaction::int IN (15, 16, 25, 26, 35, 36, 45, 46);

UPDATE preflight.acled_full
SET type_of_protest = COALESCE(type_of_protest, '') || 'y'
WHERE event_type ILIKE '%violence against civi%'
  AND interaction::int IN (15, 16);



-- We are only using events precise enough to have locations within PGM cells
-- Thus, we exclude geo_precision 3 which indicates "larger area"
-- (unclear what that means but during testing, it was nearly always ADM1 or higher.


CREATE TABLE preflight.acled AS
SELECT *
FROM preflight.acled_full
WHERE geo_precision::int < 3;



ALTER TABLE preflight.acled
    ADD PRIMARY KEY (index);
ALTER TABLE preflight.acled_full
    ADD PRIMARY KEY (index);
CREATE INDEX acled_idx ON preflight.acled (priogrid_gid, month_id, type_of_violence);
CREATE INDEX acled_full_idx ON preflight.acled_full (priogrid_gid, month_id, type_of_violence);
CREATE INDEX acled2_idx ON preflight.acled (priogrid_gid, month_id, type_of_violence, type_of_protest);
CREATE INDEX acled2_full_idx ON preflight.acled_full (priogrid_gid, month_id, type_of_violence, type_of_protest);

