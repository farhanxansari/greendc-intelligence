"use client";

import { useEffect, useState } from "react";
import { apiGet } from "./api";

type Params = Record<string, string | number | undefined>;

export function useApi<T>(path: string, params?: Params) {
  const key = JSON.stringify(params ?? {});
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    apiGet<T>(path, params)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e: Error) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, key]);

  return { data, error, loading };
}