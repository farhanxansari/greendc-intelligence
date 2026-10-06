const TZ = "America/Denver";

export const fmtDate = (iso: string) =>
  new Date(iso).toLocaleDateString("en-GB", { timeZone: TZ, day: "2-digit", month: "short", year: "numeric" });

export const fmtShort = (iso: string) =>
  new Date(iso).toLocaleDateString("en-GB", { timeZone: TZ, day: "2-digit", month: "short" });

export const fmtHour = (iso: string) =>
  new Date(iso).toLocaleString("en-GB", {
    timeZone: TZ, day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false,
  });

export const fmtDateTime = (iso: string) =>
  new Date(iso).toLocaleString("en-GB", {
    timeZone: TZ, day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });

export const fmtNum = (v: number | null | undefined, digits = 0) =>
  v == null || Number.isNaN(v)
    ? "-"
    : v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const humanize = (s: string) => s.replace(/_/g, " ").replace(/:/g, ": ");

export const fmtMonth = (iso: string) =>
  new Date(iso).toLocaleDateString("en-GB", { timeZone: TZ, month: "short", year: "2-digit" });