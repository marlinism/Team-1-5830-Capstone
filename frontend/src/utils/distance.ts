export type LatLng = {
    lat: number;
    lng: number;
};

export function haversineDistanceMeters(a: LatLng, b: LatLng): number {
    const R = 6371000;
    const toRad = (deg: number) => (deg * Math.PI) / 180;

    const dLat = toRad(b.lat - a.lat);
    const dLng = toRad(b.lng - a.lng);
    const lat1 = toRad(a.lat);
    const lat2 = toRad(b.lat);

    const sinDLat = Math.sin(dLat / 2);
    const sinDLng = Math.sin(dLng / 2);

    const h =
        sinDLat * sinDLat +
        Math.cos(lat1) * Math.cos(lat2) * sinDLng * sinDLng;

    const c = 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
    return R * c;
}

export const distanceMeters = haversineDistanceMeters;

export function formatDistance(meters: number): string {
    if (!Number.isFinite(meters)) return "-";
    if (meters < 1000) return `${Math.round(meters)} m`;
    return `${(meters / 1000).toFixed(1)} km`;
}