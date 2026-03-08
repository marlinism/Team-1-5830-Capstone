export type GeocodeResult = {
  label: string;
  lat: number;
  lng: number;
  rawFeature: any;
};

export type GeocodeSeattleOptions = {
  accessToken: string;
  mapCenter?: { lat: number; lng: number };
};

const SEATTLE_BBOX: [number, number, number, number] = [-122.4597, 47.4810, -122.2244, 47.7341];

function norm(s: string): string {
  return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
}
function tokenSet(s: string): Set<string> {
  return new Set(norm(s).split(/[^a-z0-9]+/).filter(Boolean));
}
function jaccard(a: Set<string>, b: Set<string>): number {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  a.forEach((x) => {
    if (b.has(x)) inter++;
  });
  const union = a.size + b.size - inter;
  return union ? inter / union : 0;
}

export async function geocodeSeattle(
    query: string,
    opts: GeocodeSeattleOptions
): Promise<GeocodeResult | null> {
  const q = norm(query);
  if (!q) return null;

  const { accessToken, mapCenter } = opts;

  const wantsPoi =
      /\b(university|college|school|campus|hospital|museum|stadium|center|centre|library)\b/i.test(query);

  const qTokenArr = Array.from(tokenSet(query));
  const isMultiWordQuery = qTokenArr.length >= 2;

  const proximity = mapCenter
    ? `&proximity=${encodeURIComponent(`${mapCenter.lng},${mapCenter.lat}`)}`
    : "";

  const bbox = `&bbox=${SEATTLE_BBOX[0]},${SEATTLE_BBOX[1]},${SEATTLE_BBOX[2]},${SEATTLE_BBOX[3]}`;

  const sessionToken =
    typeof crypto !== "undefined" && typeof (crypto as any).randomUUID === "function"
      ? (crypto as any).randomUUID()
      : `sess_${Math.random().toString(16).slice(2)}_${Date.now()}`;

  async function suggest(types: string): Promise<any[]> {
    const url =
      `https://api.mapbox.com/search/searchbox/v1/suggest` +
      `?q=${encodeURIComponent(query)}` +
      `&access_token=${encodeURIComponent(accessToken)}` +
      `&session_token=${encodeURIComponent(sessionToken)}` +
      `&limit=10` +
      `&types=${encodeURIComponent(types)}` +
      proximity +
      bbox;

    const res = await fetch(url);
    if (!res.ok) {
      console.warn("Mapbox searchbox suggest non-OK", res.status, res.statusText, { query, types });
      return [];
    }

    const data = await res.json();
    return Array.isArray(data?.suggestions) ? data.suggestions : [];
  }

  async function retrieve(mapboxId: string): Promise<any | null> {
    const url =
      `https://api.mapbox.com/search/searchbox/v1/retrieve/${encodeURIComponent(mapboxId)}` +
      `?access_token=${encodeURIComponent(accessToken)}` +
      `&session_token=${encodeURIComponent(sessionToken)}`;

    const res = await fetch(url);
    if (!res.ok) {
      console.warn("Mapbox searchbox retrieve non-OK", res.status, res.statusText, { mapboxId });
      return null;
    }

    const data = await res.json();
    const f = Array.isArray(data?.features) ? data.features[0] : null;
    return f || null;
  }

  let features: any[] = [];
  if (wantsPoi) {
    features = await suggest("poi");
    if (!features.length) {
      features = await suggest("poi,address,place");
    }

    const poiOnly = features.filter((s: any) => s?.feature_type === "poi" || s?.feature_type === "poi.landmark");
    if (poiOnly.length) features = poiOnly;
  } else {
    features = await suggest("poi,address,place");
  }

  if (!features.length) {
    console.warn("Mapbox searchbox: no suggestions", { query, wantsPoi });
    return null;
  }

  const qTokens = tokenSet(query);

  let bestFeature: any | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;

  features.forEach((f: any) => {
    const featureType: string = String(f?.feature_type || "");
    const isPoi = featureType.startsWith("poi");
    const isAddress = featureType === "address";
    const isPlace = featureType === "place" || featureType === "region" || featureType === "country";

    const label: string = f?.name || f?.full_address || f?.place_formatted || "";
    const text: string = f?.name || "";

    const labelTokens = tokenSet(label);
    const textTokens = tokenSet(text);

    const relevance = typeof f?.relevance === "number" ? f.relevance : (typeof f?.score === "number" ? f.score : 0);

    const tokScore = Math.max(jaccard(qTokens, labelTokens), jaccard(qTokens, textTokens));
    const phraseBonus = norm(label).includes(q) ? 0.25 : 0;

    const poiBonus = isPoi ? 0.45 : 0;

    const addressPenalty = wantsPoi && isAddress ? 0.85 : 0;

    const streetPenalty =
      wantsPoi && /\b(st|street|ave|avenue|rd|road|blvd|boulevard)\b/i.test(label) ? 0.55 : 0;

    const placePenalty = wantsPoi && isPlace ? 0.9 : 0;

    const seattleBonus = /\bseattle\b/i.test(label) ? 0.08 : 0;

    const hasUniversityToken = /\buniversity\b/i.test(query);
    const hasCollegeToken = /\bcollege\b/i.test(query);
    const wantsInstitution = wantsPoi && (hasUniversityToken || hasCollegeToken);

    const labelNorm = norm(label);
    const hasInstitutionWord = hasUniversityToken ? labelNorm.includes("university") : (hasCollegeToken ? labelNorm.includes("college") : true);

    const institutionPenalty = wantsInstitution && !hasInstitutionWord ? 1.2 : 0;
    const institutionStreetPenalty =
      wantsInstitution && hasInstitutionWord && /\b(st|street|ave|avenue|rd|road|blvd|boulevard)\b/i.test(label) ? 0.6 : 0;

    const score =
      relevance * 0.5 +
      tokScore * 0.65 +
      phraseBonus +
      poiBonus +
      seattleBonus -
      addressPenalty -
      streetPenalty -
      placePenalty -
      institutionPenalty -
      institutionStreetPenalty;

    if (score > bestScore) {
      bestScore = score;
      bestFeature = f;
    }
  });

   if (isMultiWordQuery && bestScore < 0.25) {
    return null;
  }

  const chosen = bestFeature ?? features[0];
  const mapboxId: string | undefined = chosen?.mapbox_id || chosen?.id;
  if (!mapboxId) return null;

  const full = await retrieve(mapboxId);
  if (!full) return null;

  const coords = full?.geometry?.coordinates;
  if (!Array.isArray(coords) || coords.length < 2) return null;

  const [lng, lat] = coords;
  const outLabel = full?.properties?.name || full?.properties?.full_address || full?.properties?.place_formatted || chosen?.name || query;

  return { label: outLabel, lat, lng, rawFeature: full };
}