DROP TABLE IF EXISTS spei_v2.pgm;
CREATE TABLE spei_v2.pgm AS
SELECT pgm.priogrid_gid                                                                           AS pg_id,
       pgm.month_id,
       coalesce(locf(sp1.spei_1) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_1,
       coalesce(locf(sp2.spei_2) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_2,
       coalesce(locf(sp3.spei_3) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_3,
       coalesce(locf(sp4.spei_4) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_4,
       coalesce(locf(sp5.spei_5) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_5,
       coalesce(locf(sp6.spei_6) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_6,
       coalesce(locf(sp7.spei_7) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_7,
       coalesce(locf(sp8.spei_8) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_8,
       coalesce(locf(sp9.spei_9) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0)   AS spei_9,
       coalesce(locf(sp10.spei_10) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_10,
       coalesce(locf(sp11.spei_11) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_11,
       coalesce(locf(sp12.spei_12) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_12,
       coalesce(locf(sp13.spei_13) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_13,
       coalesce(locf(sp14.spei_14) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_14,
       coalesce(locf(sp15.spei_15) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_15,
       coalesce(locf(sp16.spei_16) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_16,
       coalesce(locf(sp17.spei_17) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_17,
       coalesce(locf(sp18.spei_18) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_18,
       coalesce(locf(sp19.spei_19) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_19,
       coalesce(locf(sp20.spei_20) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_20,
       coalesce(locf(sp21.spei_21) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_21,
       coalesce(locf(sp22.spei_22) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_22,
       coalesce(locf(sp23.spei_23) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_23,
       coalesce(locf(sp24.spei_24) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_24,
       coalesce(locf(sp25.spei_25) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_25,
       coalesce(locf(sp26.spei_26) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_26,
       coalesce(locf(sp27.spei_27) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_27,
       coalesce(locf(sp28.spei_28) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_28,
       coalesce(locf(sp29.spei_29) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_29,
       coalesce(locf(sp30.spei_30) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_30,
       coalesce(locf(sp31.spei_31) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_31,
       coalesce(locf(sp32.spei_32) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_32,
       coalesce(locf(sp33.spei_33) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_33,
       coalesce(locf(sp34.spei_34) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_34,
       coalesce(locf(sp35.spei_35) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_35,
       coalesce(locf(sp36.spei_36) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_36,
       coalesce(locf(sp37.spei_37) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_37,
       coalesce(locf(sp38.spei_38) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_38,
       coalesce(locf(sp39.spei_39) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_39,
       coalesce(locf(sp40.spei_40) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_40,
       coalesce(locf(sp41.spei_41) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_41,
       coalesce(locf(sp42.spei_42) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_42,
       coalesce(locf(sp43.spei_43) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_43,
       coalesce(locf(sp44.spei_44) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_44,
       coalesce(locf(sp45.spei_45) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_45,
       coalesce(locf(sp46.spei_46) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_46,
       coalesce(locf(sp47.spei_47) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_47,
       coalesce(locf(sp48.spei_48) OVER (PARTITION BY pgm.priogrid_gid ORDER BY month_id ASC), 0) AS spei_48
FROM staging.priogrid_month AS pgm
         LEFT JOIN spei_v2.spei_1 AS sp1 ON sp1.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_2 AS sp2 ON sp2.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_3 AS sp3 ON sp3.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_4 AS sp4 ON sp4.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_5 AS sp5 ON sp5.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_6 AS sp6 ON sp6.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_7 AS sp7 ON sp7.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_8 AS sp8 ON sp8.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_9 AS sp9 ON sp9.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_10 AS sp10 ON sp10.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_11 AS sp11 ON sp11.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_12 AS sp12 ON sp12.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_13 AS sp13 ON sp13.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_14 AS sp14 ON sp14.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_15 AS sp15 ON sp15.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_16 AS sp16 ON sp16.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_17 AS sp17 ON sp17.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_18 AS sp18 ON sp18.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_19 AS sp19 ON sp19.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_20 AS sp20 ON sp20.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_21 AS sp21 ON sp21.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_22 AS sp22 ON sp22.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_23 AS sp23 ON sp23.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_24 AS sp24 ON sp24.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_25 AS sp25 ON sp25.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_26 AS sp26 ON sp26.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_27 AS sp27 ON sp27.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_28 AS sp28 ON sp28.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_29 AS sp29 ON sp29.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_30 AS sp30 ON sp30.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_31 AS sp31 ON sp31.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_32 AS sp32 ON sp32.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_33 AS sp33 ON sp33.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_34 AS sp34 ON sp34.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_35 AS sp35 ON sp35.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_36 AS sp36 ON sp36.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_37 AS sp37 ON sp37.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_38 AS sp38 ON sp38.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_39 AS sp39 ON sp39.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_40 AS sp40 ON sp40.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_41 AS sp41 ON sp41.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_42 AS sp42 ON sp42.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_43 AS sp43 ON sp43.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_44 AS sp44 ON sp44.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_45 AS sp45 ON sp45.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_46 AS sp46 ON sp46.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_47 AS sp47 ON sp47.priogrid_month_id = pgm.id
         LEFT JOIN spei_v2.spei_48 AS sp48 ON sp48.priogrid_month_id = pgm.id;