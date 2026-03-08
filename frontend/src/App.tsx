import React, { useState } from "react";
import "./App.scss";

import SideNav from "./components/Layout/SideNav";
import MapView from "./components/Map/MapView";
import NearbyListView from "./components/Nearby/NearbyListView";
import RecommendationPanel from "./components/Recommendation/RecommendationPanel";

import { useSmartParkData } from "./hooks/useSmartParkData";

import type {
    AppTab,
    Destination,
    ParkingItem,
    RecommendationPreferences,
    UserLocation,
} from "./types/parking";

const DEFAULT_PREFERENCES: RecommendationPreferences = {
    topK: 10,
    horizon: "15",
    priceLevel: "moderate",
    walkLevel: "moderate",
    safetyLevel: "high",
};

function App() {
    const [activeTab, setActiveTab] = useState<AppTab>("map");
    const [showSideNav, setShowSideNav] = useState(false);

    const [preferences, setPreferences] = useState<RecommendationPreferences>(DEFAULT_PREFERENCES);

    const [destination, setDestination] = useState<Destination>(null);
    const [currentLocation, setCurrentLocation] = useState<UserLocation>(null);
    const [selectedParking, setSelectedParking] = useState<ParkingItem | null>(null);

    const {
        items,
        loading,
        error,
        debugEnabled,
        manifest,
        selectedDate,
        selectedSlotTime,
        availableTimes,
        changeDebugDate,
        changeDebugTime,
        debugDraftDate,
        debugDraftTime,
        applyDebugSelection,
        resetDebugSelection,
    } = useSmartParkData();

    const handleChangePreferences = (patch: Partial<RecommendationPreferences>) => {
        setPreferences((prev) => ({ ...prev, ...patch }));
    };

    const handleSelectParking = (item: ParkingItem | null) => {
        setSelectedParking(item);
    };

    const openGoogleMapsDirections = async (parking: ParkingItem): Promise<void> => {
        const openWithOrigin = (lat: number, lng: number) => {
            const url =
                `https://www.google.com/maps/dir/?api=1` +
                `&origin=${lat},${lng}` +
                `&destination=${parking.lat},${parking.lng}` +
                `&travelmode=driving`;

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
                    const loc = {
                        lat: pos.coords.latitude,
                        lng: pos.coords.longitude,
                    };
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
    };

    const handleGetDirections = () => {
        if (!selectedParking) return;
        void openGoogleMapsDirections(selectedParking);
    };

    const handleTabChange = (tab: AppTab) => {
        setActiveTab(tab);
        setShowSideNav(false);
    };

    return (
        <div className={`app-shell ${debugEnabled ? "is-debug" : ""}`}>
            <button
                type="button"
                className={`app-shell__menu-btn ${showSideNav ? "is-hidden" : ""}`}
                onClick={() => setShowSideNav((prev) => !prev)}
                aria-label="Toggle navigation"
                style={debugEnabled ? { top: 62 } : undefined}
            >
                ☰
            </button>

            <SideNav
                activeTab={activeTab}
                isOpen={showSideNav}
                onChangeTab={handleTabChange}
                onClose={() => setShowSideNav(false)}
            />

            {debugEnabled && (
                <div
                    style={{
                        position: "fixed",
                        top: 0,
                        left: 0,
                        right: 0,
                        zIndex: 1400,
                        background: "#ffffff",
                        borderBottom: "1px solid #e5e7eb",
                        padding: "10px 14px",
                        display: "flex",
                        gap: "12px",
                        alignItems: "center",
                        flexWrap: "wrap",
                    }}
                >
                    <strong>DEBUG</strong>

                    <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        Date
                        <select
                            value={debugDraftDate || selectedDate}
                            onChange={(e) => changeDebugDate(e.target.value)}
                            style={{ height: 34 }}
                        >
                            {(manifest?.available_dates ?? []).map((d) => (
                                <option key={d} value={d}>
                                    {d}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        Time
                        <select
                            value={debugDraftTime || selectedSlotTime}
                            onChange={(e) => changeDebugTime(e.target.value)}
                            style={{ height: 34 }}
                        >
                            {availableTimes.map((t) => (
                                <option key={t} value={t}>
                                    {t}
                                </option>
                            ))}
                        </select>
                    </label>

                    <button
                        type="button"
                        onClick={applyDebugSelection}
                        disabled={!debugDraftDate || !debugDraftTime}
                        style={{
                            height: 34,
                            padding: "0 12px",
                            borderRadius: 10,
                            border: "1px solid #d1d5db",
                            background: "#111827",
                            color: "#fff",
                            fontWeight: 700,
                            cursor: !debugDraftDate || !debugDraftTime ? "not-allowed" : "pointer",
                            opacity: !debugDraftDate || !debugDraftTime ? 0.6 : 1,
                        }}
                    >
                        Apply & Reload
                    </button>

                    <button
                        type="button"
                        onClick={resetDebugSelection}
                        style={{
                            height: 34,
                            padding: "0 12px",
                            borderRadius: 10,
                            border: "1px solid #d1d5db",
                            background: "#fff",
                            color: "#111827",
                            fontWeight: 700,
                            cursor: "pointer",
                        }}
                    >
                        Reset
                    </button>

                    <div style={{ color: "#6b7280", fontSize: 13 }}>
                        Debug mode is enabled via <code>?debug=1</code>
                    </div>
                </div>
            )}

            {loading && <div className="app-shell__status-banner">Loading Smart Park data...</div>}

            {error && (
                <div className="app-shell__status-banner app-shell__status-banner--error">{error}</div>
            )}

            <main className="app-shell__main">
                {!loading && !error && (
                    <>
                        {activeTab === "map" ? (
                            <MapView
                                items={items}
                                destination={destination}
                                setDestination={setDestination}
                                currentLocation={currentLocation}
                                setCurrentLocation={setCurrentLocation}
                                preferences={preferences}
                                onChangePreferences={(patch) => {
                                    handleChangePreferences(patch);
                                    if (patch.horizon) setSelectedParking(null);
                                }}
                                selectedParking={selectedParking}
                                onSelectParking={handleSelectParking}
                                isSideNavOpen={showSideNav}
                            />
                        ) : (
                            <NearbyListView
                                items={items}
                                destination={destination}
                                currentLocation={currentLocation}
                                onSelectParking={(item) => {
                                    setSelectedParking(item);
                                    setActiveTab("map");
                                }}
                                onGetDirections={(item) => {
                                    setSelectedParking(item);
                                    void openGoogleMapsDirections(item);
                                }}
                            />
                        )}

                        {selectedParking && (
                            <RecommendationPanel
                                parking={selectedParking}
                                destination={destination}
                                currentLocation={currentLocation}
                                horizon={preferences.horizon}
                                onClose={() => setSelectedParking(null)}
                                onGetDirections={handleGetDirections}
                            />
                        )}
                    </>
                )}
            </main>
        </div>
    );
}

export default App;