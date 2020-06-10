## Dataset



## Transforms

ViEWS has a number of transformation functions built in.
For implementation details see the file views/apps/transforms/lib.py
where each transformation is defined in python code.

The naming convention is simple, source columns are prefixed with transformation names and parameters of the transformation.
For example: `tlag_1_ged_dummy_sb` is the time lag of 1 month of ged_dummy_sb.
Transformations can of course be chained.
For example: `time_since_greq_100_ged_best_sb` is the time since ged_best_sb (the best estimate of state-based deaths from GED) was greater or equal to 100.
Notice that order matters.
For example: `splag_1_1_time_since_ged_dummy_sb` is the first order spatial lag of time since ged_dummy_sb. This becomes a very large number as the spatial lag is the sum across the neighboring cells, which evaluates to a sum across many times_since.
This is different from `time_since_splag_1_1_ged_dummy_sb` which evaluates to the time since any neighboring cell had a ged_dummy_sb event.

### summ (sum)

Compute the sum of columns. Names should

### product (product)

### delta (delta)

### greater_or_equal (greq)

### smaller_or_equal (smeq)

### in_range (in_range)

### tlag (tlag)

### tlead (tlead)

### moving_average (ma)

### cweq (ma)

### time_since (time_since)

### decay (decay)

### mean (mean)

### ln (ln)

### demean (demean)

### rollmax (rollmax)

### onset_possible (onset_possible)

### onset (onset)

### distance_to_event (spdist)

### spacetime_distance_to_event (stdist)

### spatial_lag (splag)
