import React from "react";
import type { AppTab } from "../../types/parking";
import "./SideNav.scss";

type Props = {
    activeTab: AppTab;
    isOpen: boolean;
    onChangeTab: (tab: AppTab) => void;
    onClose: () => void;
};

export default function SideNav(
    {
        activeTab,
        isOpen,
        onChangeTab,
        onClose,
    }: Props) {
    return (
        <>
            <div
                className={`side-nav__backdrop ${isOpen ? "is-open" : ""}`}
                onClick={onClose}
            />

            <aside className={`side-nav ${isOpen ? "is-open" : ""}`}>
                <div className="side-nav__header">
                    <div className="side-nav__brand">Smart Park Seattle</div>
                    <button
                        className="side-nav__close"
                        onClick={onClose}
                        aria-label="Close menu"
                        type="button"
                    >
                        ✕
                    </button>
                </div>

                <div className="side-nav__list">
                    <button
                        type="button"
                        className={`side-nav__item ${activeTab === "map" ? "is-active" : ""}`}
                        onClick={() => {
                            onChangeTab("map");
                            onClose();
                        }}
                    >
                        Map View
                    </button>

                    <button
                        type="button"
                        className={`side-nav__item ${activeTab === "nearby" ? "is-active" : ""}`}
                        onClick={() => {
                            onChangeTab("nearby");
                            onClose();
                        }}
                    >
                        Nearby Parking
                    </button>
                </div>
            </aside>
        </>
    );
}