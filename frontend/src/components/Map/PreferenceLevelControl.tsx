import React from "react";
import "./PreferenceLevelControl.scss";

type Level = "low" | "moderate" | "high";

type Props = {
    label: string;
    value: Level;
    onChange: (value: Level) => void;
};

const OPTIONS: Level[] = ["low", "moderate", "high"];

const LABELS: Record<Level, string> = {
    low: "Low",
    moderate: "Moderate",
    high: "High",
};

const PreferenceLevelControl: React.FC<Props> = ({ label, value, onChange }) => {
    return (
        <div className="preference-level-control">
            <div className="preference-level-control__label">{label}</div>
            <div className="preference-level-control__options">
                {OPTIONS.map((option) => (
                    <button
                        key={option}
                        type="button"
                        className={`preference-level-control__chip ${
                            value === option ? "is-active" : ""
                        }`}
                        onClick={() => onChange(option)}
                    >
                        {LABELS[option]}
                    </button>
                ))}
            </div>
        </div>
    );
};

export default PreferenceLevelControl;