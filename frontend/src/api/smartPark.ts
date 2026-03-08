export type Manifest = {
    default_date: string;
    available_dates: string[];
    generated_at: string;
    version: string;
};

export type AreaRecord = {
    area_id: string;
    name?: string;
    area?: string | null;
    lat: number | null;
    lng: number | null;
    ratePerHour?: number | null;
    totalSpots?: number | null;
    riskScore?: number | null;
    riskStatus?: "Safe" | "Moderate" | "High" | "Unknown" | null;
    p15: number;
    p30: number;
};


export type SlotPayload = {
    slot_time_local: string;
    slot_time_utc: string;
    areas: AreaRecord[];
};

export type DayPayload = {
    date: string;
    generated_at: string;
    slots: SlotPayload[];
};

export type DatePayload = {
    date: string;
    slot_time_local: string;
    slot_time_utc: string;
    generated_at: string;
    areas: AreaRecord[];
};

const BASE_URL =
    "https://smart-park-seattle.s3.us-east-1.amazonaws.com/parking_v2";

export async function fetchManifest(): Promise<Manifest> {
    const url = `${BASE_URL}/serving/current/manifest.json`;
    console.log("fetch manifest url:", url);

    const res = await fetch(url, {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
    });

    if (!res.ok) {
        throw new Error(`Failed to fetch manifest: ${res.status} ${res.statusText}`);
    }

    return res.json();
}

export async function fetchLatest(): Promise<DatePayload> {
    const url = `${BASE_URL}/serving/current/latest.json`;
    console.log("fetch latest url:", url);

    const res = await fetch(url, {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
    });

    if (!res.ok) {
        throw new Error(`Failed to fetch latest: ${res.status} ${res.statusText}`);
    }

    return res.json();
}

export async function fetchByDate(date: string): Promise<DayPayload> {
    const url = `${BASE_URL}/serving/current/by-date/${date}.json`;
    console.log("fetch date url:", url);

    const res = await fetch(url, {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
    });

    if (!res.ok) {
        throw new Error(`Failed to fetch date ${date}: ${res.status} ${res.statusText}`);
    }

    return res.json();
}