

export type LatLng = { lat: number; lng: number };

export type MatrixResult = {
  distanceM: number;
  durationS: number;
};

export type WalkingMatrixOptions = {
  accessToken: string;
  signal?: AbortSignal;
  maxDestinationsPerRequest?: number;
  includeDuration?: boolean;
  includeDistance?: boolean;
};

const DEFAULT_MAX_DESTS = 24;

const cache = new Map<string, MatrixResult>();

function keyFor(source: LatLng, dest: LatLng) {
  return `${source.lat.toFixed(6)},${source.lng.toFixed(6)}|${dest.lat.toFixed(6)},${dest.lng.toFixed(6)}`;
}

function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

function isFiniteNum(x: any): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

async function fetchMatrixOnce(
  source: LatLng,
  dests: LatLng[],
  opts: WalkingMatrixOptions
): Promise<MatrixResult[]> {
  const {
    accessToken,
    signal,
    includeDistance = true,
    includeDuration = true,
  } = opts;

  const annotations: string[] = [];
  if (includeDistance) annotations.push("distance");
  if (includeDuration) annotations.push("duration");

  if (annotations.length === 0) {
    throw new Error("WalkingMatrixOptions must request distance and/or duration");
  }

  const coords = [source, ...dests]
    .map((p) => `${p.lng},${p.lat}`)
    .join(";");

  const destinations = dests.map((_, i) => String(i + 1)).join(";");

  const url =
    `https://api.mapbox.com/directions-matrix/v1/mapbox/walking/${coords}` +
    `?sources=0` +
    `&destinations=${encodeURIComponent(destinations)}` +
    `&annotations=${encodeURIComponent(annotations.join(","))}` +
    `&access_token=${encodeURIComponent(accessToken)}`;

  const res = await fetch(url, { method: "GET", signal });
  if (!res.ok) {
    throw new Error(`Mapbox Matrix failed: ${res.status} ${res.statusText}`);
  }

  const data = await res.json();

  const distRow: any[] | null = Array.isArray(data?.distances?.[0]) ? data.distances[0] : null;
  const durRow: any[] | null = Array.isArray(data?.durations?.[0]) ? data.durations[0] : null;

  return dests.map((_, i) => {
    const col = i + 1;
    const distanceM = distRow && isFiniteNum(distRow[col]) ? distRow[col] : Number.POSITIVE_INFINITY;
    const durationS = durRow && isFiniteNum(durRow[col]) ? durRow[col] : Number.POSITIVE_INFINITY;
    return { distanceM, durationS };
  });
}

/**
 * Get walking matrix results (distance + duration) from a single source to many destinations.
 */
export async function getWalkingMatrix(
  source: LatLng,
  destinations: LatLng[],
  opts: WalkingMatrixOptions
): Promise<MatrixResult[]> {
  const maxDests = opts.maxDestinationsPerRequest ?? DEFAULT_MAX_DESTS;
  const results: MatrixResult[] = new Array(destinations.length);

  const toFetchIdx: number[] = [];
  for (let i = 0; i < destinations.length; i++) {
    const k = keyFor(source, destinations[i]);
    const cached = cache.get(k);
    if (cached) {
      results[i] = cached;
    } else {
      toFetchIdx.push(i);
    }
  }

  if (toFetchIdx.length === 0) return results;

  const idxChunks = chunk(toFetchIdx, maxDests);
  for (const idxChunk of idxChunks) {
    const destChunk = idxChunk.map((i) => destinations[i]);
    const chunkRes = await fetchMatrixOnce(source, destChunk, opts);

    idxChunk.forEach((idx, j) => {
      const r = chunkRes[j];
      results[idx] = r;
      cache.set(keyFor(source, destinations[idx]), r);
    });
  }

  return results;
}

export async function getWalkingMatrixById<T extends { id: string; lat: number; lng: number }>(
  source: LatLng,
  items: T[],
  opts: WalkingMatrixOptions
): Promise<Map<string, MatrixResult>> {
  const dests = items.map((x) => ({ lat: x.lat, lng: x.lng }));
  const res = await getWalkingMatrix(source, dests, opts);
  const out = new Map<string, MatrixResult>();
  items.forEach((it, i) => out.set(it.id, res[i]));
  return out;
}
