// Backend: VITE_API_URL at build time. A full URL (the Render service), or "same-origin" when the API serves the
// built page itself (laptop + Cloudflare Tunnel). Unset: the Render service in a production build, localhost:8000
// in development (a production build must never fall back to localhost: every visitor's requests would fail).
const RAW = import.meta.env.VITE_API_URL
  || (import.meta.env.PROD ? "https://aurapath-api-vahf.onrender.com" : "http://localhost:8000");
export const API = RAW === "same-origin" ? "" : RAW.replace(/\/$/, "");

async function call(path, opts = {}, onSlow) {
  // free hosting sleeps when idle: tell the user if the first request takes a while
  const slow = onSlow ? setTimeout(() => onSlow(true), 3000) : null;
  try {
    const r = await fetch(API + path, {
      ...opts,
      headers: { "content-type": "application/json", ...(opts.headers || {}) },
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return await r.json();
  } finally {
    if (slow) clearTimeout(slow);
    if (onSlow) onSlow(false);
  }
}

export const getMeta = (onSlow) => call("/api/meta", {}, onSlow);
export const searchTowns = (q) => call("/api/towns?q=" + encodeURIComponent(q));
export const getPlace = (id) => call("/api/place/" + id);
export const getTaste = (accepted) => call("/api/taste", { method: "POST", body: JSON.stringify({ accepted }) }).then((r) => r.taste);
export const recommend =(body, onSlow) => call("/api/recommend", { method: "POST", body: JSON.stringify(body) }, onSlow);

export function sendFeedback(session, place, action) {
  call("/api/feedback", { method: "POST", body: JSON.stringify({ session, place, action }) }).catch(() => {});
}
