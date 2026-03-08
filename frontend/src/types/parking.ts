export type PreferenceLevel = "low" | "moderate" | "high";

export type RecommendationPreferences = {
    topK: 5 | 10;
    horizon: "15" | "30";
    priceLevel: PreferenceLevel;
    walkLevel: PreferenceLevel;
    safetyLevel: PreferenceLevel;
};

export type ParkingItem = {
    id: string;
    name: string;
    address?: string;
    lat: number;
    lng: number;
    ratePerHour?: number;
    p15: number;
    p30: number;
    walkDistanceM?: number;
    riskStatus?: "Safe" | "Moderate" | "High" | "Unknown";
    riskScore?: number;
    totalSpots?: number;
    area?: string;
};

export type Destination = {
    label: string;
    lat: number;
    lng: number;
} | null;

export type AppTab = "map" | "nearby";

export type UserLocation = {
    lat: number;
    lng: number;
} | null;