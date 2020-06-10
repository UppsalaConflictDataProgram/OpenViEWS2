-- Create a skeleton schema with identifiers and geographic extent only.
-- To be used down the line in joining data in pandas

DROP SCHEMA IF EXISTS skeleton CASCADE;
CREATE SCHEMA skeleton;

-- PGY
CREATE TABLE skeleton.pgy_global AS
SELECT pgy.priogrid_gid AS pg_id,
       y.year,
       cy.country_id,
       pg.in_africa,
       c.name AS country_name
FROM staging.priogrid_year AS pgy
         INNER JOIN staging.year AS y ON pgy.year_id = y.year
-- LEFT here because pgy-cy mapping stops
         LEFT JOIN staging.country_year AS cy ON pgy.country_year_id = cy.id
         INNER JOIN staging.priogrid AS pg ON pgy.priogrid_gid = pg.gid
         INNER JOIN staging.country AS c ON c.id=cy.country_id
WHERE y.year < 2031;

CREATE TABLE skeleton.pgy_africa AS
SELECT *
FROM skeleton.pgy_global
WHERE in_africa = TRUE;

-- PGM
CREATE TABLE skeleton.pgm_global AS
SELECT pgm.priogrid_gid AS pg_id,
       m.id             AS month_id,
       m.year_id        AS year,
       m.month          AS month,
       cm.country_id       country_id,
       pg.in_africa,
       c.name AS country_name
FROM staging.priogrid_month AS pgm
         INNER JOIN staging.month AS m ON m.id = pgm.month_id
         INNER JOIN staging.priogrid_year AS pgy ON pgy.year_id = m.year_id AND pgy.priogrid_gid = pgm.priogrid_gid
         INNER JOIN staging.country_month AS cm ON pgm.country_month_id = cm.id
         INNER JOIN staging.priogrid AS pg ON pg.gid = pgm.priogrid_gid
         INNER JOIN staging.country AS c ON c.id=cm.country_id
WHERE m.year_id < 2031;

CREATE TABLE skeleton.pgm_africa AS
SELECT *
FROM skeleton.pgm_global
WHERE in_africa = TRUE;

-- CY
DROP TABLE IF EXISTS skeleton.cy_global;
CREATE TABLE skeleton.cy_global AS
SELECT c.id AS country_id,
       c.in_africa,
       c.name AS country_name,
       y.year
FROM staging.country AS c
         CROSS JOIN staging.year AS y
WHERE c.gweyear = 2016
  AND y.year < 2031;

CREATE TABLE skeleton.cy_africa AS
SELECT *
FROM skeleton.cy_global
WHERE in_africa = 1;

-- CM
CREATE TABLE skeleton.cm_global AS
SELECT c.id      AS country_id,
       m.year_id AS year,
       m.id      AS month_id,
       m.month,
       c.name AS country_name,
       c.in_africa
FROM staging.country AS c
         CROSS JOIN staging.month AS m
WHERE c.gweyear = 2016
  AND m.year_id < 2031;

CREATE TABLE skeleton.cm_africa AS
SELECT *
FROM skeleton.cm_global
WHERE in_africa = 1;

