export function getSeattleNowParts() {
    const now = new Date();

    const formatter = new Intl.DateTimeFormat("en-CA", {
        timeZone: "America/Los_Angeles",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
    });

    const parts = formatter.formatToParts(now);
    const map = Object.fromEntries(
        parts.filter((p) => p.type !== "literal").map((p) => [p.type, p.value])
    );

    return {
        year: Number(map.year),
        month: Number(map.month),
        day: Number(map.day),
        hour: Number(map.hour),
        minute: Number(map.minute),
        date: `${map.year}-${map.month}-${map.day}`,
    };
}

export function getSeattleDisplayTime() {
    return new Intl.DateTimeFormat("en-US", {
        timeZone: "America/Los_Angeles",
        dateStyle: "medium",
        timeStyle: "short",
    }).format(new Date());
}

export function getNextQuarterHourMinute(minute: number): number {
    return Math.ceil(minute / 15) * 15;
}

export function getNextTargetSlotIso(
    horizon: "15" | "30"
): string {
    const now = new Date();
    const seattle = getSeattleNowParts();

    let hour = seattle.hour;
    let minute = getNextQuarterHourMinute(seattle.minute);

    if (minute === 60) {
        minute = 0;
        hour += 1;
    }

    if (horizon === "30") {
        minute += 15;
        if (minute >= 60) {
            minute -= 60;
            hour += 1;
        }
    }

    const hh = String(hour).padStart(2, "0");
    const mm = String(minute).padStart(2, "0");

    return `${seattle.date}T${hh}:${mm}`;
}

export function findBestSlotIndex(
    slotTimes: string[],
    targetPrefix: string
): number {
    const exact = slotTimes.findIndex((t) => t.startsWith(targetPrefix));
    if (exact >= 0) return exact;
    return 0;
}