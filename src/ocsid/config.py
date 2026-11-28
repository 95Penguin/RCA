"""Fixed RE2 OCSID–CPCA v4 profile; core tree APIs retain strict defaults.

These are method parameters, not settings selected separately per dataset.
The 1ms clock bound reflects the measured timestamp quantization assumption.
"""
ATTRIBUTION_DEFAULTS = {
    "min_op_requests": 1,
    "max_requests": 100,
    "temperature": 1.,
    "aggregation": "uniform",
    "restoration": "replace",
    "gap_policy": "post",
    "clock_tolerance_us": 1000.,
    "tree_policy": "entry-component",
}
