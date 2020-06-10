CREATE TABLE acled.cm AS
    SELECT cm.month_id,
           cm.country_id,
           coalesce(cm.acled_count_pr, 0) AS acled_count_pr,
           coalesce(cm.acled_count_sb, 0) AS acled_count_sb,
           coalesce(cm.acled_count_ns, 0) AS acled_count_ns,
           coalesce(cm.acled_count_os, 0) AS acled_count_os
FROM staging.country_month AS cm;

CREATE TABLE acled.pgm AS
    SELECT pgm.month_id,
           pgm.priogrid_gid AS pg_id,
           coalesce(pgm.acled_count_pr, 0) AS acled_count_pr,
           coalesce(pgm.acled_count_sb, 0) AS acled_count_sb,
           coalesce(pgm.acled_count_ns, 0) AS acled_count_ns,
           coalesce(pgm.acled_count_os, 0) AS acled_count_os,
           coalesce(pgm.acled_fat_sb, 0) AS acled_fat_sb,
           coalesce(pgm.acled_fat_ns, 0) AS acled_fat_ns,
           coalesce(pgm.acled_fat_os, 0) AS acled_fat_os,
           coalesce(pgm.acled_fat_pr, 0) AS acled_fat_pr
FROM staging.priogrid_month AS pgm;