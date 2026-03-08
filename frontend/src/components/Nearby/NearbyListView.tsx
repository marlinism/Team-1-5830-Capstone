import React, { useMemo, useState } from "react";
import type { Destination, ParkingItem, UserLocation } from "../../types/parking";
import { haversineDistanceMeters, formatDistance } from "../../utils/distance";
import "./NearbyListView.scss";

type Props = {
    items: ParkingItem[];
    destination: Destination;
    currentLocation: UserLocation;
    onSelectParking?: (item: ParkingItem) => void;
    onGetDirections?: (item: ParkingItem) => void;
};

type SortBy = "nearest_me" | "best_p15" | "best_p30" | "lowest_rate";

function formatSafetyLabel(riskStatus?: string, riskScore?: number): string {
    if (riskStatus && riskStatus.trim().length > 0) return riskStatus;

    if (riskScore == null || !Number.isFinite(riskScore)) return "Unknown";
    if (riskScore < 0.34) return "Safe";
    if (riskScore < 0.67) return "Moderate";
    return "High";
}

const NearbyListView: React.FC<Props> = ({
                                             items,
                                             destination,
                                             currentLocation,
                                             onSelectParking,
                                             onGetDirections,
                                         }) => {
    const [topK, setTopK] = useState<5 | 10>(5);
    const [sortBy, setSortBy] = useState<SortBy>("nearest_me");
    const [areaFilter, setAreaFilter] = useState<string>("all");

    const areas = useMemo(() => {
        const vals = Array.from(
            new Set(items.map((x) => x.area).filter(Boolean) as string[])
        ).sort();
        return vals;
    }, [items]);

    const visibleItems = useMemo(() => {
        const filtered =
            areaFilter === "all" ? items : items.filter((p) => p.area === areaFilter);

        const enriched = filtered.map((p) => {
            const distToDestination =
                destination != null
                    ? haversineDistanceMeters(
                        { lat: p.lat, lng: p.lng },
                        { lat: destination.lat, lng: destination.lng }
                    )
                    : Number.POSITIVE_INFINITY;

            const distToMe =
                currentLocation != null
                    ? haversineDistanceMeters(
                        { lat: p.lat, lng: p.lng },
                        { lat: currentLocation.lat, lng: currentLocation.lng }
                    )
                    : Number.POSITIVE_INFINITY;

            return {
                ...p,
                _distanceToDestination: distToDestination,
                _distanceToMe: distToMe,
            };
        });

        enriched.sort((a, b) => {
            switch (sortBy) {
                case "nearest_me":
                    return (a._distanceToMe ?? Number.POSITIVE_INFINITY) - (b._distanceToMe ?? Number.POSITIVE_INFINITY);
                case "best_p15":
                    return (b.p15 ?? 0) - (a.p15 ?? 0);
                case "best_p30":
                    return (b.p30 ?? 0) - (a.p30 ?? 0);
                case "lowest_rate":
                    return (a.ratePerHour ?? Number.POSITIVE_INFINITY) - (b.ratePerHour ?? Number.POSITIVE_INFINITY);
                default:
                    return 0;
            }
        });

        return enriched.slice(0, topK);
    }, [items, areaFilter, sortBy, topK, destination, currentLocation]);

    return (
        <section className="nearby-list-view">
            <div className="nearby-list-view__header">
                <div>
                    <h2>Nearby Parking</h2>
                    <p>Top 5 / Top 10 nearby parking with filters and sorting</p>
                </div>

                <div className="nearby-list-view__filters">
                    <label>
                        Show
                        <select
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value) as 5 | 10)}
                        >
                            <option value={5}>Top 5</option>
                            <option value={10}>Top 10</option>
                        </select>
                    </label>

                    <label>
                        Sort by
                        <select value={sortBy} onChange={(e) => setSortBy(e.target.value as SortBy)}>
                            <option value="nearest_me">Nearest to me</option>
                            <option value="best_p15">Best success (15m)</option>
                            <option value="best_p30">Best success (30m)</option>
                            <option value="lowest_rate">Lowest rate</option>
                        </select>
                    </label>

                    <label>
                        Area
                        <select value={areaFilter} onChange={(e) => setAreaFilter(e.target.value)}>
                            <option value="all">All areas</option>
                            {areas.map((a) => (
                                <option key={a} value={a}>
                                    {a}
                                </option>
                            ))}
                        </select>
                    </label>
                </div>
            </div>

            <div className="nearby-list-view__list">
                {visibleItems.map((p) => {
                    const addr = (p.address ?? "").trim();
                    const nameNorm = (p.name ?? "").trim().replace(/\s+/g, " ").toUpperCase();
                    const addrNorm = addr.replace(/\s+/g, " ").toUpperCase();
                    const showAddress = addrNorm.length > 0 && addrNorm !== nameNorm;

                    return (
                        <div key={p.id} className="nearby-card">
                            <div
                                className="nearby-card__main"
                                role="button"
                                tabIndex={0}
                                onClick={() => onSelectParking?.(p)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter" || e.key === " ") {
                                        e.preventDefault();
                                        onSelectParking?.(p);
                                    }
                                }}
                            >
                                <div className="nearby-card__left">
                                    <div className="nearby-card__name">{p.name}</div>
                                    {showAddress ? (
                                        <div className="nearby-card__address">{addr}</div>
                                    ) : null}

                                    <div className="nearby-card__meta">
                                        {p.area ? <span>{p.area}</span> : null}
                                        {Number.isFinite((p as any)._distanceToDestination) ? (
                                            <span>
                                                {p.area ? " • " : ""}Walk:{" "}
                                                {formatDistance((p as any)._distanceToDestination)}
                                            </span>
                                        ) : null}
                                        {Number.isFinite((p as any)._distanceToMe) ? (
                                            <span>
                                                {(p.area || Number.isFinite((p as any)._distanceToDestination))
                                                    ? " • "
                                                    : ""}To me:{" "}
                                                {formatDistance((p as any)._distanceToMe)}
                                            </span>
                                        ) : null}
                                    </div>
                                </div>

                                <div className="nearby-card__right">
                                    <div className="nearby-card__stats">
                                        <div className="nearby-card__stat">
                                            <span>P15</span>
                                            <strong>{Math.round((p.p15 ?? 0) * 100)}%</strong>
                                        </div>
                                        <div className="nearby-card__stat">
                                            <span>P30</span>
                                            <strong>{Math.round((p.p30 ?? 0) * 100)}%</strong>
                                        </div>
                                        <div className="nearby-card__stat">
                                            <span>Rate</span>
                                            <strong>
                                                {p.ratePerHour != null ? `$${p.ratePerHour}/hr` : "N/A"}
                                            </strong>
                                        </div>
                                        <div className="nearby-card__stat">
                                            <span>Safety</span>
                                            <strong>
                                                {formatSafetyLabel((p as any).riskStatus, (p as any).riskScore)}
                                            </strong>
                                        </div>

                                        <button
                                            type="button"
                                            className="nearby-card__direction-btn"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onGetDirections?.(p);
                                            }}
                                        >
                                            Get Directions
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </section>
    );
};

export default NearbyListView;