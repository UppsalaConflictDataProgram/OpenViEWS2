-- Drop existing attached
DROP TABLE IF EXISTS preflight.ged_attached_full;
DROP TABLE IF EXISTS preflight.ged_attached;

-- Create preflight.ged_attached
CREATE TABLE preflight.ged_attached AS
    (
        WITH month_ged AS
                 (
                     SELECT *,
                            EXTRACT(MONTH FROM date_start :: DATE) AS month_start,
                            EXTRACT(MONTH FROM date_end :: DATE)   AS month_end
                     FROM dataprep.ged
                 ),
             month_ged_start AS
                 (
                     SELECT month_ged.*,
                            staging.month.id AS month_id_start
                     FROM month_ged,
                          staging.month
                     WHERE (month_ged.year :: INT = staging.month.year_id AND
                            month_ged.month_start = staging.month.month)
                 ),
             month_ged_full AS
                 (
                     SELECT month_ged_start.*,
                            staging.month.id AS month_id_end
                     FROM month_ged_start,
                          staging.month
                     WHERE (month_ged_start.year :: INT = staging.month.year_id AND
                            month_ged_start.month_end = staging.month.month)
                 )
        SELECT *
        FROM month_ged_full
    );

-- Add ids
ALTER TABLE preflight.ged_attached ADD PRIMARY KEY (id);
ALTER TABLE preflight.ged_attached ADD COLUMN country_month_id_end bigint;
ALTER TABLE preflight.ged_attached ADD COLUMN country_month_id_start bigint;
ALTER TABLE preflight.ged_attached DROP COLUMN IF EXISTS geom;
ALTER TABLE preflight.ged_attached ADD COLUMN geom geometry(point, 4326);
UPDATE preflight.ged_attached
SET geom=st_setsrid(st_geometryfromtext(geom_wkt), 4326)
WHERE geom_wkt <> '';

-- Create preflight.ged_attached_full
CREATE TABLE preflight.ged_attached_full AS SELECT * FROM preflight.ged_attached;


DELETE FROM preflight.ged_attached WHERE where_prec IN (4,6,7);
ALTER TABLE preflight.ged_attached_full ADD PRIMARY KEY (id);
CREATE INDEX ged_attached_gidx ON preflight.ged_attached USING GIST(geom);
CREATE INDEX ged_attached_idx ON preflight.ged_attached (priogrid_gid,month_id_end, type_of_violence);
CREATE INDEX ged_attached_s_idx ON preflight.ged_attached (priogrid_gid,month_id_start, type_of_violence);
CREATE INDEX ged_attached_full_gidx ON preflight.ged_attached_full USING GIST(geom);
CREATE INDEX ged_attached_fullx_s_idx ON preflight.ged_attached_full (priogrid_gid,month_id_end, type_of_violence);
CREATE INDEX ged_attached_fullx_gidx ON preflight.ged_attached_full (priogrid_gid,month_id_start, type_of_violence);


-- Update preflight.ged_attached_full
WITH a AS
         (SELECT cm.*, c.gwcode
          FROM staging.country_month cm
                   LEFT JOIN
               staging.country c ON (cm.country_id = c.id))
UPDATE preflight.ged_attached_full
SET country_month_id_end=a.id
FROM a
WHERE (a.gwcode = ged_attached_full.country_id AND a.month_id = ged_attached_full.month_id_end);
WITH a AS
         (SELECT cm.*, c.gwcode
          FROM staging.country_month cm
                   LEFT JOIN
               staging.country c ON (cm.country_id = c.id))
UPDATE preflight.ged_attached_full
SET country_month_id_start=a.id
FROM a
WHERE (a.gwcode = ged_attached_full.country_id AND a.month_id = ged_attached_full.month_id_start);