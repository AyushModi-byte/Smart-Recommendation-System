// Accounts (Supabase). Off unless VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY are set at build time.
// Schema: supabase/schema.sql, then supabase/002_people.sql.
import { createClient } from "@supabase/supabase-js";

const URL_ = import.meta.env.VITE_SUPABASE_URL;
const KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;
export const sb = URL_ && KEY ? createClient(URL_, KEY) : null;
export const accountsEnabled = !!sb;

const redirect = () => ({ emailRedirectTo: window.location.origin });
const ok = ({ data, error }) => { if (error) throw error; return data; };

// ---------- auth ----------
export const signInPassword = (email, password) => sb.auth.signInWithPassword({ email, password }).then(ok);
export const signUpPassword = (email, password) => sb.auth.signUp({ email, password, options: redirect() }).then(ok);
export const sendMagicLink = (email) => sb.auth.signInWithOtp({ email, options: redirect() }).then(ok);
export const resetPassword = (email) => sb.auth.resetPasswordForEmail(email, { redirectTo: window.location.origin }).then(ok);
export const updatePassword = (password) => sb.auth.updateUser({ password }).then(ok);
export const signOut = () => sb.auth.signOut();

// ---------- profile ----------
export const loadProfile = (uid) => sb.from("profiles").select("*").eq("id", uid).maybeSingle().then(ok);

export async function saveProfile(uid, p) {
  const row = { id: uid, name: p.name, home: p.home, moods: p.moods };
  if (p.discoverable !== undefined) row.discoverable = p.discoverable;
  return sb.from("profiles").upsert(row).select().single().then(ok);
}

// ---------- likes, saves, skips ----------
// -> {accept: [card…], reject: [id…], save: [card…]}, oldest first
export async function loadActions() {
  const data = await sb.from("place_actions").select("place_id, action, card, created_at").order("created_at").then(ok);
  const out = { accept: [], reject: [], save: [] };
  for (const r of data) out[r.action].push(r.action === "reject" ? r.place_id : { ...(r.card || {}), id: r.place_id });
  return out;
}

export async function putActions(rows) {                 // rows: [{place_id, action, card}]
  if (rows.length) await sb.from("place_actions").upsert(rows, { onConflict: "user_id,place_id,action" }).then(ok);
}

export const deleteAction = (place_id, action) =>
  sb.from("place_actions").delete().eq("place_id", place_id).eq("action", action).then(ok);

export const slimCard = (c) => ({ id: c.id, name: c.name, town: c.town, country: c.country, category: c.category,
                                  lat: c.lat, lon: c.lon, photo: c.photo });

// ---------- posts ----------
export const addPost = (card, body) =>
  sb.from("posts").insert({ place_id: card.id, place: slimCard(card), body }).select("id, body, created_at, place").single().then(ok);

export const myPosts = (uid) =>
  sb.from("posts").select("id, body, created_at, place").eq("user_id", uid).order("created_at", { ascending: false }).limit(100).then(ok);

export const communityPosts = () =>
  sb.from("posts").select("id, body, created_at, place, profiles(name)").order("created_at", { ascending: false }).limit(40).then(ok);

// ---------- people with similar taste ----------
export const putTaste = (row) => sb.from("tastes").upsert(row).then(ok);
export const deleteTaste = (uid) => sb.from("tastes").delete().eq("user_id", uid).then(ok);

function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) s += a[i] * b[i];
  return s;                                              // both are unit vectors
}

// -> [{user_id, name, match (0-100), favorites, liked_count, top_categories}], best first
export async function similarUsers(uid, taste, k = 12) {
  const rows = await sb.from("tastes").select("user_id, taste, favorites, liked_count, top_categories, profiles(name)")
    .neq("user_id", uid).order("updated_at", { ascending: false }).limit(1000).then(ok);
  return rows.map((r) => ({ ...r, name: r.profiles?.name || "Someone", sim: cosine(taste, r.taste) }))
    .sort((a, b) => b.sim - a.sim).slice(0, k)
    .map((r) => ({ ...r, match: Math.round(50 * (r.sim + 1)) }));
}
