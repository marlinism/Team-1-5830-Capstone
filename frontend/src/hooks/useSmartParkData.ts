import { useEffect, useMemo, useState } from "react";
import {
  fetchManifest,
  fetchLatest,
  fetchByDate,
  Manifest,
  DatePayload,
  DayPayload,
} from "../api/smartPark";
import type { ParkingItem } from "../types/parking";

export function useSmartParkData() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [datePayload, setDatePayload] = useState<DatePayload | null>(null);
  const [dayPayload, setDayPayload] = useState<DayPayload | null>(null);
  const [debugEnabled, setDebugEnabled] = useState<boolean>(false);
  const [availableTimes, setAvailableTimes] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [selectedSlotTime, setSelectedSlotTime] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const DEBUG_STORAGE_KEY = "smartpark_debug_selection";

  const readDebugSelection = (): { date?: string; time?: string } => {
    try {
      const raw = localStorage.getItem(DEBUG_STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  };

  const writeDebugSelection = (sel: { date: string; time: string }) => {
    localStorage.setItem(DEBUG_STORAGE_KEY, JSON.stringify(sel));
  };

  const clearDebugSelection = () => {
    localStorage.removeItem(DEBUG_STORAGE_KEY);
  };

  useEffect(() => {
    async function init() {
      try {
        setLoading(true);
        setError(null);

        const manifestData = await fetchManifest();
        setManifest(manifestData);

        const params = new URLSearchParams(window.location.search);
        const isDebug = params.get("debug") === "1";
        const stored = readDebugSelection();

        setDebugEnabled(isDebug);

        if (isDebug) {
          const useDate =
            stored.date ||
            manifestData.default_date ||
            (manifestData.available_dates?.[0] ?? "");
          if (!useDate) {
            throw new Error("No available dates in manifest.");
          }

          const day = await fetchByDate(useDate);
          setDayPayload(day);
          setDatePayload(null);
          setSelectedDate(day.date);

          const times = day.slots.map((s) => s.slot_time_local.slice(11, 16));
          setAvailableTimes(times);

          const picked =
            (stored.time && day.slots.find((s) => s.slot_time_local.slice(11, 16) === stored.time)) ||
            day.slots[day.slots.length - 1];

          setSelectedSlotTime(picked ? picked.slot_time_local.slice(11, 16) : "");
        } else {
          const latest = await fetchLatest();
          setDayPayload(null);
          setDatePayload(latest);
          setSelectedDate(latest.date);
          setSelectedSlotTime(latest.slot_time_local.slice(11, 16));
          setAvailableTimes([]);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
      } finally {
        setLoading(false);
      }
    }

    init();
  }, []);

  const items: ParkingItem[] = useMemo(() => {
    if (!debugEnabled) {
      if (!datePayload) return [];
      return datePayload.areas.map((a) => ({
        id: a.area_id,
        name: a.name ?? a.area_id,
        address: a.name ?? a.area_id,
        lat: a.lat ?? 0,
        lng: a.lng ?? 0,
        ratePerHour: a.ratePerHour ?? undefined,
        p15: a.p15 ?? 0,
        p30: a.p30 ?? 0,
        riskStatus: a.riskStatus ?? "Unknown",
        riskScore: a.riskScore ?? undefined,
        totalSpots: a.totalSpots ?? undefined,
        area: a.area ?? undefined,
      }));
    }

    if (!dayPayload) return [];
    const slot =
      dayPayload.slots.find((s) => s.slot_time_local.slice(11, 16) === selectedSlotTime) ||
      dayPayload.slots[0];
    if (!slot) return [];

    return slot.areas.map((a) => ({
      id: a.area_id,
      name: a.name ?? a.area_id,
      address: a.name ?? a.area_id,
      lat: a.lat ?? 0,
      lng: a.lng ?? 0,
      ratePerHour: a.ratePerHour ?? undefined,
      p15: a.p15 ?? 0,
      p30: a.p30 ?? 0,
      riskStatus: a.riskStatus ?? "Unknown",
      riskScore: a.riskScore ?? undefined,
      totalSpots: a.totalSpots ?? undefined,
      area: a.area ?? undefined,
    }));
  }, [debugEnabled, datePayload, dayPayload, selectedSlotTime]);

  const [debugDraftDate, setDebugDraftDate] = useState<string>("");
  const [debugDraftTime, setDebugDraftTime] = useState<string>("");

  useEffect(() => {
    if (!debugEnabled) return;
    setDebugDraftDate(selectedDate);
    setDebugDraftTime(selectedSlotTime);
  }, [debugEnabled, selectedDate, selectedSlotTime]);

  const applyDebugSelection = () => {
    if (!debugEnabled) return;
    if (!debugDraftDate || !debugDraftTime) return;
    writeDebugSelection({ date: debugDraftDate, time: debugDraftTime });
    window.location.reload();
  };

  const resetDebugSelection = () => {
    if (!debugEnabled) return;
    clearDebugSelection();
    window.location.reload();
  };

  const changeDebugDate = (nextDate: string) => {
    if (!debugEnabled) return;
    setDebugDraftDate(nextDate);
    if (nextDate !== selectedDate) {
      setDebugDraftTime("");
    }
  };

  const changeDebugTime = (hhmm: string) => {
    if (!debugEnabled) return;
    setDebugDraftTime(hhmm);
  };

  return {
    manifest,
    datePayload,
    dayPayload,
    debugEnabled,
    selectedDate,
    selectedSlotTime,
    availableTimes,
    items,
    loading,
    error,
    changeDebugDate,
    changeDebugTime,
    debugDraftDate,
    debugDraftTime,
    applyDebugSelection,
    resetDebugSelection,
  };
}