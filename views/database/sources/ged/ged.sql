CREATE TABLE ged.cm AS
SELECT cm.month_id,
       cm.country_id,
       cm.ged_best_sb,
       cm.ged_best_ns,
       cm.ged_best_os,
       cm.ged_count_sb,
       cm.ged_count_ns,
       cm.ged_count_os
FROM staging.country_month AS cm;

CREATE TABLE ged.pgm_unimp
AS
SELECT pgm.month_id,
       pgm.priogrid_gid                  AS pg_id,
       pgm.ged_best_sb,
       pgm.ged_best_ns,
       pgm.ged_best_os,
       pgm.ged_count_sb,
       pgm.ged_count_ns,
       pgm.ged_count_os,
       public.to_dummy(pgm.ged_count_sb) AS ged_dummy_sb,
       public.to_dummy(pgm.ged_count_ns) AS ged_dummy_ns,
       public.to_dummy(pgm.ged_count_os) AS ged_dummy_os
FROM staging.priogrid_month AS pgm;

CREATE TABLE ged.pgm_geoimp_0
AS
SELECT pgm.month_id,
       pgm.priogrid_gid       AS pg_id,
       pgm.ged_best_sb,
       pgm.ged_best_ns,
       pgm.ged_best_os,
       pgm.ged_count_sb,
       pgm.ged_count_ns,
       pgm.ged_count_os,
       pgm_imp.ged_sb_dummy_1 AS ged_dummy_sb,
       pgm_imp.ged_ns_dummy_1 AS ged_dummy_ns,
       pgm_imp.ged_os_dummy_1 AS ged_dummy_os
FROM staging.priogrid_month AS pgm
         LEFT JOIN left_imputation.pgm AS pgm_imp
                   ON pgm_imp.priogrid_gid = pgm.priogrid_gid AND pgm_imp.month_id = pgm.month_id;

CREATE TABLE ged.pgm_geoimp_1
AS
SELECT pgm.month_id,
       pgm.priogrid_gid       AS pg_id,
       pgm.ged_best_sb,
       pgm.ged_best_ns,
       pgm.ged_best_os,
       pgm.ged_count_sb,
       pgm.ged_count_ns,
       pgm.ged_count_os,
       pgm_imp.ged_sb_dummy_2 AS ged_dummy_sb,
       pgm_imp.ged_ns_dummy_2 AS ged_dummy_ns,
       pgm_imp.ged_os_dummy_2 AS ged_dummy_os
FROM staging.priogrid_month AS pgm
         LEFT JOIN left_imputation.pgm AS pgm_imp
                   ON pgm_imp.priogrid_gid = pgm.priogrid_gid AND pgm_imp.month_id = pgm.month_id;

CREATE TABLE ged.pgm_geoimp_2
AS
SELECT pgm.month_id,
       pgm.priogrid_gid       AS pg_id,
       pgm.ged_best_sb,
       pgm.ged_best_ns,
       pgm.ged_best_os,
       pgm.ged_count_sb,
       pgm.ged_count_ns,
       pgm.ged_count_os,
       pgm_imp.ged_sb_dummy_3 AS ged_dummy_sb,
       pgm_imp.ged_ns_dummy_3 AS ged_dummy_ns,
       pgm_imp.ged_os_dummy_3 AS ged_dummy_os
FROM staging.priogrid_month AS pgm
         LEFT JOIN left_imputation.pgm AS pgm_imp
                   ON pgm_imp.priogrid_gid = pgm.priogrid_gid AND pgm_imp.month_id = pgm.month_id;

CREATE TABLE ged.pgm_geoimp_3
AS
SELECT pgm.month_id,
       pgm.priogrid_gid       AS pg_id,
       pgm.ged_best_sb,
       pgm.ged_best_ns,
       pgm.ged_best_os,
       pgm.ged_count_sb,
       pgm.ged_count_ns,
       pgm.ged_count_os,
       pgm_imp.ged_sb_dummy_4 AS ged_dummy_sb,
       pgm_imp.ged_ns_dummy_4 AS ged_dummy_ns,
       pgm_imp.ged_os_dummy_4 AS ged_dummy_os
FROM staging.priogrid_month AS pgm
         LEFT JOIN left_imputation.pgm AS pgm_imp
                   ON pgm_imp.priogrid_gid = pgm.priogrid_gid AND pgm_imp.month_id = pgm.month_id;

CREATE TABLE ged.pgm_geoimp_4
AS
SELECT pgm.month_id,
       pgm.priogrid_gid       AS pg_id,
       pgm.ged_best_sb,
       pgm.ged_best_ns,
       pgm.ged_best_os,
       pgm.ged_count_sb,
       pgm.ged_count_ns,
       pgm.ged_count_os,
       pgm_imp.ged_sb_dummy_5 AS ged_dummy_sb,
       pgm_imp.ged_ns_dummy_5 AS ged_dummy_ns,
       pgm_imp.ged_os_dummy_5 AS ged_dummy_os
FROM staging.priogrid_month AS pgm
         LEFT JOIN left_imputation.pgm AS pgm_imp
                   ON pgm_imp.priogrid_gid = pgm.priogrid_gid AND pgm_imp.month_id = pgm.month_id;
