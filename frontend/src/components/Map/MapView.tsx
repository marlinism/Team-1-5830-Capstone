import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import mapboxgl, { Marker, Popup } from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import ReactDOMServer from "react-dom/server";

import type {
    Destination,
    ParkingItem,
    RecommendationPreferences,
    UserLocation,
} from "../../types/parking";
import { haversineDistanceMeters } from "../../utils/distance";
import ParkingPopupCard from "./ParkingPopupCard";
import PreferenceLevelControl from "./PreferenceLevelControl";
import SearchResultsSheet from "./SearchResultsSheet";
import "./MapView.scss";

import { geocodeSeattle } from "../../utils/geocode";
import { getWalkingMatrixById, type MatrixResult } from "../../api/mapboxMatrix";

type Props = {
    items: ParkingItem[];
    destination: Destination;
    setDestination: (d: Destination) => void;
    currentLocation: UserLocation;
    setCurrentLocation: (loc: UserLocation) => void;
    preferences: RecommendationPreferences;
    onChangePreferences: (patch: Partial<RecommendationPreferences>) => void;
    selectedParking: ParkingItem | null;
    onSelectParking: (item: ParkingItem | null) => void;
    isSideNavOpen: boolean;
};

const MAPBOX_TOKEN =
    (process.env.REACT_APP_MAPBOX_TOKEN as string | undefined) || undefined;

const DEFAULT_CENTER = { lat: 47.6205, lng: -122.3493 };
const MAX_VISIBLE_MARKERS = 200;

type PreferenceLevel = "low" | "moderate" | "high";

function levelToTarget(value: PreferenceLevel): number {
    switch (value) {
        case "low":
            return 0;
        case "moderate":
            return 0.5;
        case "high":
            return 1;
        default:
            return 0.5;
    }
}

function clamp01(x: number): number {
    return Math.max(0, Math.min(1, x));
}

function maxPriceForLevel(level: PreferenceLevel): number {
    switch (level) {
        case "low":
            return 6.5;
        case "moderate":
            return 10;
        case "high":
        default:
            return Number.POSITIVE_INFINITY;
    }
}

function minAvailabilityForLevel(level: PreferenceLevel): number {
    switch (level) {
        case "low":
            return 0;
        case "moderate":
            return 0.5;
        case "high":
        default:
            return 0.7;
    }
}

function maxRiskForSafetyLevel(level: PreferenceLevel): number {
    switch (level) {
        case "low":
            return 1;
        case "moderate":
            return 0.66;
        case "high":
        default:
            return 0.33;
    }
}

function levelIndex(level: PreferenceLevel): number {
    switch (level) {
        case "low":
            return 0;
        case "moderate":
            return 1;
        case "high":
        default:
            return 2;
    }
}

function bandScore(selected: PreferenceLevel, actual: PreferenceLevel): number {
    const diff = Math.abs(levelIndex(selected) - levelIndex(actual));
    if (diff === 0) return 1;
    if (diff === 1) return 0.66;
    return 0.33;
}

function priceBand(ratePerHour: number | null | undefined): PreferenceLevel {
    const r = typeof ratePerHour === "number" && Number.isFinite(ratePerHour)
        ? ratePerHour
        : Number.POSITIVE_INFINITY;
    if (r <= 6.5) return "low";
    if (r <= 10) return "moderate";
    return "high";
}

function availabilityBand(successScore: number): PreferenceLevel {
    if (!Number.isFinite(successScore)) return "moderate";
    if (successScore < 0.5) return "low";
    if (successScore < 0.7) return "moderate";
    return "high";
}

function safetyBand(riskScore: number | null | undefined): PreferenceLevel {
    const r = typeof riskScore === "number" && Number.isFinite(riskScore)
        ? clamp01(riskScore)
        : 0.5;
    if (r < 0.33) return "low"; // Safe
    if (r < 0.66) return "moderate";
    return "high";
}

const MapView: React.FC<Props> = ({
                                      items,
                                      destination,
                                      setDestination,
                                      currentLocation,
                                      setCurrentLocation,
                                      preferences,
                                      onChangePreferences,
                                      selectedParking,
                                      onSelectParking,
                                      isSideNavOpen,
                                  }) => {
    const mapContainerRef = useRef<HTMLDivElement | null>(null);
    const mapRef = useRef<mapboxgl.Map | null>(null);
    const markersRef = useRef<Marker[]>([]);
    const userMarkerRef = useRef<Marker | null>(null);
    const markerElsRef = useRef<Record<string, HTMLButtonElement>>({});
    const popupByIdRef = useRef<Record<string, Popup>>({});
    const activePopupRef = useRef<Popup | null>(null);
    const lastAutoFocusedIdRef = useRef<string | null>(null);

    const [showFinder, setShowFinder] = useState(false);
    const [searchText, setSearchText] = useState("");
    const [visibleItems, setVisibleItems] = useState<ParkingItem[]>([]);
    const [showResultsSheet, setShowResultsSheet] = useState(false);

    const [walkingMatrixById, setWalkingMatrixById] = useState<
        Map<string, MatrixResult>
    >(new Map());
    const [walkingMatrixLoading, setWalkingMatrixLoading] = useState(false);

    const openGoogleMapsDirections = useCallback(
        async (parking: ParkingItem): Promise<void> => {
            const openWithOrigin = (lat: number, lng: number) => {
                const url =
                    `https://www.google.com/maps/dir/?api=1` +
                    `&origin=${lat},${lng}` +
                    `&destination=${parking.lat},${parking.lng}` +
                    `&travelmode=walking`;

                window.open(url, "_blank", "noopener,noreferrer");
            };

            if (currentLocation) {
                openWithOrigin(currentLocation.lat, currentLocation.lng);
                return;
            }

            if (!navigator.geolocation) {
                const fallbackUrl =
                    `https://www.google.com/maps/search/?api=1&query=${parking.lat},${parking.lng}`;
                window.open(fallbackUrl, "_blank", "noopener,noreferrer");
                return;
            }

            return new Promise((resolve) => {
                navigator.geolocation.getCurrentPosition(
                    (pos) => {
                        const loc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                        setCurrentLocation(loc);
                        openWithOrigin(loc.lat, loc.lng);
                        resolve();
                    },
                    () => {
                        const fallbackUrl =
                            `https://www.google.com/maps/search/?api=1&query=${parking.lat},${parking.lng}`;
                        window.open(fallbackUrl, "_blank", "noopener,noreferrer");
                        resolve();
                    },
                    { enableHighAccuracy: true, timeout: 8000 }
                );
            });
        },
        [currentLocation, setCurrentLocation]
    );

    const attachPopupHandlers = useCallback(
        (popup: Popup, parking: ParkingItem) => {
            window.setTimeout(() => {
                const popupEl = popup.getElement();
                const directionBtn = popupEl?.querySelector(
                    '[data-role="popup-direction-btn"]'
                ) as HTMLButtonElement | null;

                if (directionBtn) {
                    directionBtn.onclick = (e) => {
                        e.stopPropagation();
                        onSelectParking(parking);
                        void openGoogleMapsDirections(parking);
                    };
                }
            }, 0);
        },
        [onSelectParking, openGoogleMapsDirections]
    );

    const searchResults = useMemo(() => {
        if (!destination) return [];


        const ranked = items.map((p) => {
            const successScore =
                preferences.horizon === "15" ? (p.p15 ?? 0) : (p.p30 ?? 0);

            const riskNorm = p.riskScore != null ? clamp01(p.riskScore) : 0.5;

            const priceCap = maxPriceForLevel(preferences.priceLevel);
            const minAvail = minAvailabilityForLevel(preferences.walkLevel);
            const maxRisk = maxRiskForSafetyLevel(preferences.safetyLevel);

            const rate = p.ratePerHour ?? Number.POSITIVE_INFINITY;
            if (rate > priceCap) return null;

            if (successScore < minAvail) return null;

            if (p.riskScore != null && riskNorm > maxRisk) return null;


            const pBand = priceBand(p.ratePerHour);
            const aBand = availabilityBand(successScore);
            const sBand = safetyBand(p.riskScore);

            const preferenceScore =
                (bandScore(preferences.priceLevel, pBand) +
                    bandScore(preferences.walkLevel, aBand) +
                    bandScore(preferences.safetyLevel, sBand)) /
                3;

            const matrix = walkingMatrixById.get(p.id);
            const walkDistanceM = matrix?.distanceM;
            const walkDurationS = matrix?.durationS;

            const distanceToDestination =
                typeof walkDistanceM === "number" && Number.isFinite(walkDistanceM)
                    ? walkDistanceM
                    : haversineDistanceMeters(
                        { lat: p.lat, lng: p.lng },
                        { lat: destination.lat, lng: destination.lng }
                    );

            if (
                typeof walkDurationS === "number" &&
                Number.isFinite(walkDurationS) &&
                walkDurationS > 18 * 60
            ) {
                return null;
            }

            const MAX_DIST_M = 1500;
            if (!Number.isFinite(distanceToDestination) || distanceToDestination > MAX_DIST_M) {
                return null;
            }

            const distanceScore = 1 / (1 + distanceToDestination / 400);

            const timeScore =
                typeof walkDurationS === "number" && Number.isFinite(walkDurationS)
                    ? 1 / (1 + walkDurationS / (8 * 60))
                    : 0.5;

            const totalScore =
                distanceScore * 0.6 +
                timeScore * 0.05 +
                successScore * 0.2 +
                preferenceScore * 0.15;

            return {
                ...p,
                _recommendationScore: totalScore,
                _distanceToDestination: distanceToDestination,
            };
        });

        const filtered = ranked.filter(Boolean) as Array<
            ParkingItem & { _recommendationScore: number; _distanceToDestination: number }
        >;

        filtered.sort((a, b) => b._recommendationScore - a._recommendationScore);
        return filtered.slice(0, 10);
    }, [items, destination, preferences, walkingMatrixById]);

    useEffect(() => {
        if (!destination) {
            setWalkingMatrixById(new Map());
            return;
        }
        const token = MAPBOX_TOKEN as string | undefined;
        if (!token) return;
        const accessToken: string = token;

        const destLat = destination.lat;
        const destLng = destination.lng;

        let cancelled = false;

        async function run() {
            try {
                setWalkingMatrixLoading(true);

                // Pre-filter by haversine to keep request small.
                const candidates = items
                    .map((p) => {
                        const d = haversineDistanceMeters(
                            { lat: p.lat, lng: p.lng },
                            { lat: destLat, lng: destLng }
                        );
                        return { id: p.id, lat: p.lat, lng: p.lng, _d: d };
                    })
                    .filter((p) => Number.isFinite(p._d) && p._d <= 8000)
                    .sort((a, b) => a._d - b._d)
                    .slice(0, 80);

                const distMap = await getWalkingMatrixById(
                    { lat: destLat, lng: destLng },
                    candidates.map((c) => ({ id: c.id, lat: c.lat, lng: c.lng })),
                    { accessToken }
                );

                if (!cancelled) setWalkingMatrixById(distMap);
            } catch {
                if (!cancelled) setWalkingMatrixById(new Map());
            } finally {
                if (!cancelled) setWalkingMatrixLoading(false);
            }
        }

        run();
        return () => {
            cancelled = true;
        };
    }, [destination, items]);

    const markerItems = useMemo(() => {
        const merged = new Map<string, ParkingItem>();
        visibleItems.forEach((p) => merged.set(p.id, p));
        searchResults.forEach((p) => merged.set(p.id, p));
        if (selectedParking) merged.set(selectedParking.id, selectedParking);
        return Array.from(merged.values());
    }, [visibleItems, searchResults, selectedParking]);

    const updateVisibleItems = useCallback(() => {
        const map = mapRef.current;
        if (!map) return;

        const bounds = map.getBounds();
        if (!bounds) return;

        const filtered = items
            .filter((p) => bounds.contains([p.lng, p.lat]))
            .sort((a, b) =>
                preferences.horizon === "15" ? (b.p15 ?? 0) - (a.p15 ?? 0) : (b.p30 ?? 0) - (a.p30 ?? 0)
            )
            .slice(0, MAX_VISIBLE_MARKERS);

        setVisibleItems(filtered);
    }, [items, preferences.horizon]);

    useEffect(() => {
        if (!MAPBOX_TOKEN) return;
        mapboxgl.accessToken = MAPBOX_TOKEN;
    }, []);

    useEffect(() => {
        if (!MAPBOX_TOKEN) return;
        if (!mapContainerRef.current) return;
        if (mapRef.current) return;

        const initialCenter = currentLocation ?? DEFAULT_CENTER;

        const map = new mapboxgl.Map({
            container: mapContainerRef.current,
            style: "mapbox://styles/mapbox/streets-v12",
            center: [initialCenter.lng, initialCenter.lat],
            zoom: 13,
        });

        map.addControl(new mapboxgl.NavigationControl(), "bottom-right");
        mapRef.current = map;

        return () => {
            activePopupRef.current?.remove();
            activePopupRef.current = null;

            markersRef.current.forEach((m) => m.remove());
            markersRef.current = [];
            markerElsRef.current = {};
            popupByIdRef.current = {};
            userMarkerRef.current?.remove();
            userMarkerRef.current = null;
            map.remove();
            mapRef.current = null;
        };
    }, [currentLocation]);

    useEffect(() => {
        if (currentLocation) return;
        if (!navigator.geolocation) return;

        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const loc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                setCurrentLocation(loc);

                if (mapRef.current) {
                    mapRef.current.flyTo({
                        center: [loc.lng, loc.lat],
                        zoom: 14,
                        essential: true,
                    });
                }
            },
            () => {

            },
            { enableHighAccuracy: true, timeout: 8000 }
        );
    }, [currentLocation, setCurrentLocation]);

    useEffect(() => {
        const map = mapRef.current;
        if (!map || !currentLocation) return;

        if (userMarkerRef.current) {
            userMarkerRef.current.setLngLat([currentLocation.lng, currentLocation.lat]);
            return;
        }

        const el = document.createElement("div");
        el.className = "user-location-marker";

        userMarkerRef.current = new mapboxgl.Marker({ element: el, anchor: "center" })
            .setLngLat([currentLocation.lng, currentLocation.lat])
            .addTo(map);
    }, [currentLocation]);

    useEffect(() => {
        const map = mapRef.current;
        if (!map) return;

        const handler = () => updateVisibleItems();

        map.on("load", handler);
        map.on("moveend", handler);
        map.on("zoomend", handler);

        handler();

        return () => {
            map.off("load", handler);
            map.off("moveend", handler);
            map.off("zoomend", handler);
        };
    }, [updateVisibleItems]);

    useEffect(() => {
        const map = mapRef.current;
        if (!map) return;

        markersRef.current.forEach((m) => m.remove());
        markersRef.current = [];
        markerElsRef.current = {};
        popupByIdRef.current = {};

        markerItems.forEach((p) => {
            const el = document.createElement("button");
            el.type = "button";
            el.className = "parking-marker";
            el.setAttribute("aria-label", p.name);

            markerElsRef.current[p.id] = el;

            const popupHtml = ReactDOMServer.renderToString(<ParkingPopupCard parking={p} />);

            const popup = new Popup({
                offset: 20,
                maxWidth: "320px",
                anchor: "top",
                closeOnClick: true,
            }).setHTML(popupHtml);

            popupByIdRef.current[p.id] = popup;

            const marker = new mapboxgl.Marker({ element: el, anchor: "center" })
                .setLngLat([p.lng, p.lat])
                .addTo(map);

            el.addEventListener("click", (evt) => {
                evt.stopPropagation();

                if (activePopupRef.current && activePopupRef.current !== popup) {
                    activePopupRef.current.remove();
                }

                popup.setLngLat([p.lng, p.lat]).addTo(map);
                activePopupRef.current = popup;

                lastAutoFocusedIdRef.current = null;
                onSelectParking(p);
                attachPopupHandlers(popup, p);

                popup.on("close", () => {
                    if (activePopupRef.current === popup) {
                        activePopupRef.current = null;
                    }
                });
            });

            markersRef.current.push(marker);
        });
    }, [markerItems, onSelectParking, attachPopupHandlers]);

    useEffect(() => {
        Object.entries(markerElsRef.current).forEach(([id, el]) => {
            if (selectedParking && id === selectedParking.id) el.classList.add("is-selected");
            else el.classList.remove("is-selected");
        });
    }, [selectedParking]);

    useEffect(() => {
        const map = mapRef.current;
        if (!map) return;

        if (!selectedParking) {
            lastAutoFocusedIdRef.current = null;
            if (activePopupRef.current) {
                activePopupRef.current.remove();
                activePopupRef.current = null;
            }
            return;
        }

        if (lastAutoFocusedIdRef.current === selectedParking.id) return;
        lastAutoFocusedIdRef.current = selectedParking.id;

        map.flyTo({
            center: [selectedParking.lng, selectedParking.lat],
            zoom: Math.max(map.getZoom(), 16),
            essential: true,
        });

        const popup = popupByIdRef.current[selectedParking.id];
        if (popup) {
            if (activePopupRef.current && activePopupRef.current !== popup) {
                activePopupRef.current.remove();
            }
            popup.setLngLat([selectedParking.lng, selectedParking.lat]).addTo(map);
            activePopupRef.current = popup;
            attachPopupHandlers(popup, selectedParking);
        }
    }, [selectedParking, attachPopupHandlers]);

    useEffect(() => {
        if (isSideNavOpen) {
            setShowFinder(false);
            setShowResultsSheet(false);
        }
    }, [isSideNavOpen]);

    const handleSearchSubmit = async () => {
        if (!MAPBOX_TOKEN || !searchText.trim()) return;
        const map = mapRef.current;
        if (!map) return;

        try {
            const center = map.getCenter();

            const token = MAPBOX_TOKEN as string | undefined;
            if (!token) return;

            let result = await geocodeSeattle(searchText, {
                accessToken: token,
                mapCenter: { lat: center.lat, lng: center.lng },
            });

            // Fallback: if the first attempt returns null, try appending "Seattle, WA".
            if (!result) {
                result = await geocodeSeattle(`${searchText}, Seattle, WA`, {
                    accessToken: token,
                    mapCenter: { lat: center.lat, lng: center.lng },
                });
            }

            if (!result) {
                console.warn("GEOCODE RESULT: null (no match)");
                return;
            }

            const q = searchText.trim().toLowerCase();
            const qTokens = q.split(/[^a-z0-9]+/).filter(Boolean);
            const labelLower = (result.label || "").toLowerCase();
            const placeTypes: string[] = Array.isArray(result.rawFeature?.place_type)
                ? result.rawFeature.place_type
                : [];

            const looksLikePoiQuery = qTokens.length >= 2;
            const isPlaceLevel = placeTypes.includes("place") || placeTypes.includes("region") || placeTypes.includes("country");
            const isAddress = placeTypes.includes("address");

            const hasAllTokens = looksLikePoiQuery && qTokens.every((t) => labelLower.includes(t));

            const streety = /\b(st|street|ave|avenue|rd|road|blvd|boulevard)\b/i.test(result.label || "");
            const institutiony = /\b(university|college|school|hospital|museum|park)\b/i.test(q);

            if (looksLikePoiQuery) {
                if (isPlaceLevel) {
                    console.warn("GEOCODE REJECTED (place-level):", result.label, placeTypes);
                    return;
                }
                if (!hasAllTokens) {
                    console.warn("GEOCODE REJECTED (missing tokens):", result.label, { qTokens, placeTypes });
                    return;
                }
                if (institutiony && (isAddress || streety) && !placeTypes.includes("poi")) {
                    console.warn("GEOCODE REJECTED (street/address for POI query):", result.label, placeTypes);
                    return;
                }
            }

            console.log(
                "GEOCODE RESULT:",
                result.label,
                placeTypes,
                result.rawFeature?.id
            );

            setDestination({ lat: result.lat, lng: result.lng, label: result.label });
            setShowResultsSheet(true);
            setShowFinder(false);

            map.flyTo({
                center: [result.lng, result.lat],
                zoom: 15,
                essential: true,
            });
        } catch (e) {
            console.error("Geocoding failed:", e);
        }
    };

    if (!MAPBOX_TOKEN) {
        return (
            <div className="map-view map-view--error">
                <div className="map-view__error-card">
                    <h3>Missing Mapbox token</h3>
                    <p>
                        Add <code>REACT_APP_MAPBOX_TOKEN</code> to your <code>.env</code> file.
                    </p>
                    <p>Example:</p>
                    <pre>REACT_APP_MAPBOX_TOKEN=pk....</pre>
                </div>
            </div>
        );
    }

    return (
        <section className="map-view">
            <div ref={mapContainerRef} className="map-view__canvas" />

            {!isSideNavOpen && (
                <div className="map-view__controls">
                    {!showFinder ? (
                        <button
                            type="button"
                            className="map-view__find-btn"
                            onClick={() => setShowFinder(true)}
                        >
                            Find parking
                        </button>
                    ) : (
                        <div className="map-view__finder-card">
                            <div className="map-view__finder-row">
                                <input
                                    className="map-view__search-input"
                                    type="text"
                                    placeholder="Search place or address (e.g., Seattle University)"
                                    value={searchText}
                                    onChange={(e) => setSearchText(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === "Enter") handleSearchSubmit();
                                    }}
                                />
                                <button type="button" onClick={handleSearchSubmit}>
                                    Search
                                </button>
                                <button type="button" onClick={() => setShowFinder(false)}>
                                    ✕
                                </button>
                            </div>

                            <div className="map-view__finder-section">
                                <div className="map-view__section-title">Recommendation preference</div>

                                <label>
                                    Parking in the next
                                    <select
                                        value={preferences.horizon}
                                        onChange={(e) =>
                                            onChangePreferences({ horizon: e.target.value as "15" | "30" })
                                        }
                                    >
                                        <option value="15">15 min</option>
                                        <option value="30">30 min</option>
                                    </select>
                                </label>

                                <PreferenceLevelControl
                                    label="Price"
                                    value={preferences.priceLevel}
                                    onChange={(value) => onChangePreferences({ priceLevel: value })}
                                />

                                <PreferenceLevelControl
                                    label="Parking availability"
                                    value={preferences.walkLevel}
                                    onChange={(value) => onChangePreferences({ walkLevel: value })}
                                />

                                <PreferenceLevelControl
                                    label="Safety preference"
                                    value={preferences.safetyLevel}
                                    onChange={(value) => onChangePreferences({ safetyLevel: value })}
                                />
                            </div>

                            {walkingMatrixLoading ? (
                                <div style={{ padding: "10px 12px", color: "#6b7280", fontSize: "0.85rem" }}>
                                    Computing walking distances…
                                </div>
                            ) : null}
                        </div>
                    )}
                </div>
            )}

            <SearchResultsSheet
                isOpen={showResultsSheet}
                items={searchResults}
                destination={destination}
                currentLocation={currentLocation}
                horizon={preferences.horizon}
                onClose={() => setShowResultsSheet(false)}
                onSelectParking={(item) => {
                    setShowResultsSheet(false);
                    setShowFinder(false);
                    onSelectParking(item);
                }}
                onGetDirections={(item) => {
                    onSelectParking(item);
                    void openGoogleMapsDirections(item);
                }}
            />
        </section>
    );
};

export default MapView;