import re

def _norm(s: str) -> str:
    """
    Normalize columns name

    Parameter: Column name
    Returns: columnname without space or special character
    """
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

ALIASES = {
    "ts": ["OccupancyDateTime", "occupancydatetime", "ts"],
    "paid_occupancy": ["PaidOccupancy", "paidoccupancy", "paid_occupancy"],
    "space_count": ["ParkingSpaceCount", "parkingspacecount", "space_count"],
    "sourceelementkey": ["SourceElementKey", "sourceelementkey"],
    "location": ["Location", "location"],
    "area": ["PaidParkingArea", "paidparkingarea", "area"],
    "subarea": ["PaidParkingSubArea", "paidparkingsubarea", "subarea"],
    "rate": ["PaidParkingRate", "paidparkingrate", "rate"],
    "parking_cat": ["ParkingCategory", "parkingcategory", "parking_cat"],
    "blockfacename": ["BlockfaceName", "blockfacename"],
    "sideofstreet": ["SideOfStreet", "sideofstreet"],
    "time_limit_cat": ["ParkingTimeLimitCategory", "parkingtimelimitcategory", "time_limit_cat"],
}

REQUIRED = {"ts", "paid_occupancy", "space_count", "sourceelementkey", "location"}

def build_rename_map(header_cols) -> dict:
    """
    Build rename map from raw CSV header -> canonical names.
    """
    norm_to_actual = {_norm(c): c for c in header_cols}
    rename_map = {}

    for canon, variants in ALIASES.items():
        for v in variants:
            key = _norm(v)
            if key in norm_to_actual:
                rename_map[norm_to_actual[key]] = canon
                break

    missing = [c for c in REQUIRED if c not in rename_map.values()]
    if missing:
        raise KeyError(
            f"Missing required columns after resolve: {missing}\n"
            f"Header columns: {list(header_cols)}"
        )

    return rename_map

def canonical_usecols(header_cols) -> list[str]:
    """
    Return the subset of header column names needed to compute canonical fields.
    """
    rename_map = build_rename_map(header_cols)
    return list(rename_map.keys())