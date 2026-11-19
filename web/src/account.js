// Optional accounts (Supabase). Off unless VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY are set at build time;
// then likes, saves, skips and posts sync to the user's account and posts are shared. Schema: supabase/schema.sql.
import { createClient } from "@supabase/supabase-js";

const URL_ = import.meta.env.VITE_SUPABASE_URL;
const KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;
export const sb = URL_ && KEY ? createClient(URL_, KEY) : null;
export const accountsEnabled = !!sb;

export async function sendMagicLink(email) {
  const { error } = await sb.auth.signInWithOtp({ email, options: { emailRedirectTo: window.location.origin } });
  if (error) throw error;
}

export const signOut = () => sb.auth.signOut();

export async function loadProfile(uid) {
  const { data, error } = await sb.from("profiles").select("*").eq("id", uid).maybeSingle();
  if (error) throw error;
  return data;
}

export async function saveProfile(uid, p) {
  const { data, error } = await sb.from("profiles").upsert({ id: uid, name: p.name, home: p.home, moods: p.moods }).select().single();
  if (error) throw error;
  return data;
}

// -> {accept: [card…], reject: [id…], save: [card…]}, oldest first
export async function loadActions() {
  const { data, error } = await sb.from("place_actions").select("place_id, action, card, created_at").order("created_at");
  if (error) throw error;
  const out = { accept: [], reject: [], save: [] };
  for (const r of data) out[r.action].push(r.action === "reject" ? r.place_id : { ...(r.card || {}), id: r.place_id });
  return out;
}

export async function putActions(rows) {                 // rows: [{place_id, action, card}]
  if (!rows.length) return;
  const { error } = await sb.from("place_actions").upsert(rows, { onConflict: "user_id,place_id,action" });
  if (error) throw error;
}

export async function deleteAction(place_id, action) {
  const { error } = await sb.from("place_actions").delete().eq("place_id", place_id).eq("action", action);
  if (error) throw error;
}

const slimCard = (c) => ({ id: c.id, name: c.name, town: c.town, country: c.country, category: c.category,
                            lat: c.lat, lon: c.lon, photo: c.photo });

export async function addPost(card, body) {
  const { data, error } = await sb.from("posts").insert({ place_id: card.id, place: slimCard(card), body })
    .select("id, body, created_at, place").single();
  if (error) throw error;
  return data;
}

export async function myPosts(uid) {
  const { data, error } = await sb.from("posts").select("id, body, created_at, place").eq("user_id", uid)
    .order("created_at", { ascending: false }).limit(100);
  if (error) throw error;
  return data;
}

export async function communityPosts() {
  const { data, error } = await sb.from("posts").select("id, body, created_at, place, profiles(name)")
    .order("created_at", { ascending: false }).limit(40);
  if (error) throw error;
  return data;
}

export { slimCard };
