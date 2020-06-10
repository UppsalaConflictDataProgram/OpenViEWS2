-- Create a grid similar to priogrid but with 1x1 degree resolution
-- Priogrid is 0.5x0.5 degree resolution.
-- SPEI comies in at 1x1 resolution so we use this to map SPEI to pg_ids

CREATE OR REPLACE FUNCTION
ST_CreateFishnet(
    -- PARAMETERS
        nrow integer, ncol integer,
        ysize float8, xsize float8,
        y0 float8 DEFAULT 0, x0 float8 DEFAULT 0,
        srid integer DEFAULT 4326,
        OUT "row" integer, OUT col integer,
        OUT geom geometry)
    -- RETURNS
        RETURNS SETOF record AS
    -- PROCESS
        $$
        SELECT i + 1 AS row, j + 1 AS col, ST_SetSRID(ST_Translate(cell, j * $3 + $5, i * $4 + $6), $7) AS geom
        FROM generate_series(0, $1 - 1) AS j,
             generate_series(0, $2 - 1) AS i,
             (SELECT ('POLYGON((0 0, 0 '||$4||', '||$3||' '||$4||', '||$3||' 0,0 0))')::geometry AS cell) AS foo;
        $$ LANGUAGE sql IMMUTABLE STRICT;


-- Create global 1x1 grid
DROP TABLE IF EXISTS spei_v2.unigrid_world;
CREATE TABLE spei_v2.unigrid_world (
    gid serial NOT NULL,
    "row" integer,
    col integer,
    cell geometry(Polygon, 4326),
    CONSTRAINT unigrid_pkey PRIMARY KEY (gid));
INSERT INTO spei_v2.unigrid_world ("row", col, cell) SELECT * FROM ST_CreateFishnet(360, 180, 1.0, 1.0, -180, -90, 4326) AS cells;
CREATE INDEX ON spei_v2.unigrid_world USING GIST (cell);


-- Create table of pg_ids to ug_ids
DROP TABLE IF EXISTS spei_v2.pg_ug;
CREATE TABLE spei_v2.pg_ug AS
SELECT pg.gid AS pg_id,
       ug.gid AS ug_id
FROM staging.priogrid AS pg,
     spei_v2.unigrid_world as ug
--Returns true if no point in pg.geom is outside of ug.cell, otherwise false.
-- ug.cells cover pg.geometries
WHERE ST_Covers(ug.cell, pg.geom);

DROP TABLE spei_v2.unigrid_world;