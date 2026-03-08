import React from "react";
import type { Destination, ParkingItem, UserLocation } from "../../types/parking";
import { formatDistance, haversineDistanceMeters } from "../../utils/distance";
import "./RecommendationPanel.scss";

type Props = {
  parking: ParkingItem;
  destination: Destination;
  currentLocation: UserLocation;
  horizon: "15" | "30";
  onGetDirections: () => void;
  onClose: () => void;
};

const RecommendationPanel: React.FC<Props> = ({
                                                parking,
                                                destination,
                                                currentLocation,
                                                horizon,
                                                onGetDirections,
                                                onClose,
                                              }) => {
  const successProb = horizon === "15" ? parking.p15 : parking.p30;

  const distanceToDestination =
      destination != null
          ? haversineDistanceMeters(
              { lat: parking.lat, lng: parking.lng },
              { lat: destination.lat, lng: destination.lng }
          )
          : null;

  const distanceToMe =
      currentLocation != null
          ? haversineDistanceMeters(
              { lat: parking.lat, lng: parking.lng },
              { lat: currentLocation.lat, lng: currentLocation.lng }
          )
          : null;

  return (
      <aside className="recommendation-panel">
        <div className="recommendation-panel__header">
          <div>
            <div className="recommendation-panel__title">{parking.name}</div>
            <div className="recommendation-panel__subtitle">{parking.address}</div>
          </div>
          <button
              type="button"
              className="recommendation-panel__close"
              onClick={onClose}
              aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="recommendation-panel__grid">
          <div>
            <div className="label">Success ({horizon}m)</div>
            <div className="value">{Math.round(successProb * 100)}%</div>
          </div>

          <div>
            <div className="label">Rate</div>
            <div className="value">
              {parking.ratePerHour != null ? `$${parking.ratePerHour}/hr` : "N/A"}
            </div>
          </div>

          <div>
            <div className="label">Risk</div>
            <div className="value">
              {parking.riskScore != null ? parking.riskScore.toFixed(2) : "N/A"}
            </div>
          </div>

          <div>
            <div className="label">Total spots</div>
            <div className="value">{parking.totalSpots ?? "N/A"}</div>
          </div>

          {distanceToDestination != null && (
              <div>
                <div className="label">To destination</div>
                <div className="value">{formatDistance(distanceToDestination)}</div>
              </div>
          )}

          {distanceToMe != null && (
              <div>
                <div className="label">To current location</div>
                <div className="value">{formatDistance(distanceToMe)}</div>
              </div>
          )}
        </div>

        <button
            type="button"
            className="recommendation-panel__cta"
            onClick={onGetDirections}
        >
          Get Directions
        </button>
      </aside>
  );
};

export default RecommendationPanel;