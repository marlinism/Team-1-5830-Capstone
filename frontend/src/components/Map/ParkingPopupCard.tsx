import React from "react";
import type { ParkingItem } from "../../types/parking";
import "./ParkingPopupCard.scss";

type Props = {
    parking: ParkingItem;
};

const ParkingPopupCard: React.FC<Props> = ({ parking }) => {
    return (
        <div className="parking-popup-card">
            <div className="parking-popup-card__title">{parking.name}</div>

            {parking.address ? (
                <div className="parking-popup-card__address">{parking.address}</div>
            ) : null}

            {parking.area ? (
                <div className="parking-popup-card__area">{parking.area}</div>
            ) : null}

            <div className="parking-popup-card__grid">
                <div>
                    <div className="label">P(15 min)</div>
                    <div className="value">{Math.round((parking.p15 ?? 0) * 100)}%</div>
                </div>

                <div>
                    <div className="label">P(30 min)</div>
                    <div className="value">{Math.round((parking.p30 ?? 0) * 100)}%</div>
                </div>

                <div>
                    <div className="label">Rate</div>
                    <div className="value">
                        {parking.ratePerHour != null ? `$${parking.ratePerHour}/hr` : "N/A"}
                    </div>
                </div>

                <div>
                    <div className="label">Risk</div>
                    <div className="value">{parking.riskStatus ?? "Unknown"}</div>
                </div>
            </div>

            <button
                type="button"
                className="parking-popup-card__direction-btn"
                data-role="popup-direction-btn"
            >
                Get Directions
            </button>
        </div>
    );
};

export default ParkingPopupCard;