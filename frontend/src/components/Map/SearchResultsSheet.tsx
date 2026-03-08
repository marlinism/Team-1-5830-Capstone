import React from "react";
import type { Destination, ParkingItem, UserLocation } from "../../types/parking";
import { haversineDistanceMeters, formatDistance } from "../../utils/distance";
import "./SearchResultsSheet.scss";

type Props = {
    isOpen: boolean;
    items: ParkingItem[];
    destination: Destination;
    currentLocation: UserLocation;
    horizon: "15" | "30";
    onClose: () => void;
    onSelectParking: (item: ParkingItem) => void;
    onGetDirections: (item: ParkingItem) => void;
};

const SearchResultsSheet: React.FC<Props> = ({
                                                 isOpen,
                                                 items,
                                                 destination,
                                                 currentLocation,
                                                 horizon,
                                                 onClose,
                                                 onSelectParking,
                                                 onGetDirections,
                                             }) => {
    if (!isOpen) return null;

    const enriched = items.map((p) => {
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

    return (
        <>
            <div className="search-results-sheet__backdrop" onClick={onClose} />

            <section className="search-results-sheet">
                <div className="search-results-sheet__handle" />

                <div className="search-results-sheet__header">
                    <div>
                        <h3>Search results</h3>
                        <p>Top {items.length} recommendations</p>
                    </div>

                    <button type="button" onClick={onClose} aria-label="Close results">
                        ✕
                    </button>
                </div>

                <div className="search-results-sheet__list">
                    {enriched.map((p) => {
                        const walkLabel = Number.isFinite((p as any)._distanceToDestination)
                            ? formatDistance((p as any)._distanceToDestination)
                            : "N/A";

                        return (
                            <div key={p.id} className="search-results-card">
                                <button
                                    type="button"
                                    className="search-results-card__main"
                                    onClick={() => onSelectParking(p)}
                                >
                                    <div className="search-results-card__line">
                                        <div className="search-results-card__name">{p.name}</div>
                                        <div className="search-results-card__walk">
                                            Walking distance: <strong>{walkLabel}</strong>
                                        </div>
                                    </div>
                                </button>

                                <button
                                    type="button"
                                    className="search-results-card__direction-btn"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onGetDirections(p);
                                    }}
                                >
                                    Get Directions
                                </button>
                            </div>
                        );
                    })}
                </div>
            </section>
        </>
    );
};

export default SearchResultsSheet;