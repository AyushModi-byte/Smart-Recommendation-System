// Account screens: sign-in by email link, first-time profile, profile tab, community feed (see account.js).
import { useEffect, useState } from "react";
import * as acct from "./account.js";

const initial = (profile, user) => (profile?.name || user?.email || "?").slice(0, 1).toUpperCase();

export function Profile({ user, profile, liked, posts, onSignOut, onMood, onHome, onEdit }) {
  const counts = {};
  for (const c of liked) counts[c.category] = (counts[c.category] || 0) + 1;
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3);
  return (
    <div className="profile">
      <div className="profile-head">
        <span className="avatar big">{initial(profile, user)}</span>
        <span><b>{profile?.name || "New user"}</b><small>{user.email}</small></span>
      </div>
      <div className="stats"><span><b>{liked.length}</b> liked</span><span><b>{posts.length}</b> posts</span></div>
      {top.length > 0 && <p>Your taste: {top.map(([k, n]) => `${k} ${Math.round((100 * n) / liked.length)}%`).join(", ")}.</p>}
      {profile?.home && <button className="link" onClick={onHome}>Show places around {profile.home.name}</button>}
      {profile?.moods?.length > 0 && (
        <div className="chips wrap">{profile.moods.map((m) => <button key={m} className="chip" onClick={() => onMood(m)}>{m}</button>)}</div>
      )}
      <div className="compose-foot">
        <button className="btn ghost" onClick={onEdit}>Edit profile</button>
        <button className="btn ghost" onClick={onSignOut}>Sign out</button>
      </div>
    </div>
  );
}

export function Community() {
  const [rows, setRows] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { acct.communityPosts().then(setRows).catch((e) => setErr(e.message)); }, []);
  if (err) return <p className="muted">Could not load posts. {err}</p>;
  if (!rows) return <p className="muted">Loading…</p>;
  if (!rows.length) return <p className="muted">No posts yet. Like a place and say something about it.</p>;
  return rows.map((p) => (
    <div key={p.id} className="post">
      <div className="post-place">{p.place?.name} <small>{p.place?.town}</small></div>
      <p>{p.body}</p>
      <small>{p.profiles?.name || "Someone"} · {new Date(p.created_at).toLocaleString()}</small>
    </div>
  ));
}

export function SignIn({ onClose }) {
  const [email, setEmail] = useState("");
  const [state, setState] = useState("");
  const submit = async (e) => {
    e.preventDefault();
    setState("sending");
    try { await acct.sendMagicLink(email.trim()); setState("sent"); } catch (err) { setState("error: " + err.message); }
  };
  return (
    <div className="overlay" onClick={onClose}>
      <form className="compose" onClick={(e) => e.stopPropagation()} onSubmit={submit} role="dialog" aria-label="Sign in">
        <div className="compose-head">Sign in to keep your likes on every device and share posts</div>
        {state === "sent" ? <p>Check your inbox: a sign-in link is on its way to <b>{email}</b>.</p> : (
          <input className="field" type="email" required autoFocus placeholder="you@example.com" value={email}
                 onChange={(e) => setEmail(e.target.value)} aria-label="Email address" />
        )}
        {state.startsWith("error") && <small className="danger">Could not send the link ({state.slice(7)}).</small>}
        <div className="compose-foot">
          <button type="button" className="btn ghost" onClick={onClose}>Close</button>
          {state !== "sent" && <button type="submit" className="btn" disabled={state === "sending"}>Email me a link</button>}
        </div>
      </form>
    </div>
  );
}

export function Onboarding({ profile, area, moods, onSave, onClose }) {
  const [name, setName] = useState(profile?.name || "");
  const [picked, setPicked] = useState(profile?.moods || []);
  const [useArea, setUseArea] = useState(!profile?.home);
  const [err, setErr] = useState("");
  const toggle = (m) => setPicked((p) => (p.includes(m) ? p.filter((x) => x !== m) : [...p, m]));
  const submit = async (e) => {
    e.preventDefault();
    try {
      await onSave({ name: name.trim(), moods: picked,
                     home: useArea ? { name: area.name, lat: area.lat, lon: area.lon, radius: area.radius } : profile?.home || null });
    } catch (x) { setErr(x.message); }
  };
  return (
    <div className="overlay" onClick={onClose || undefined}>
      <form className="compose" onClick={(e) => e.stopPropagation()} onSubmit={submit} role="dialog" aria-label="Your profile">
        <div className="compose-head">{profile ? "Edit your profile" : "Welcome to AuraPath"}</div>
        <label className="field-label">Name
          <input className="field" required maxLength={40} value={name} autoFocus onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={useArea} onChange={(e) => setUseArea(e.target.checked)} />
          Make {area.name} my home area{profile?.home && !useArea ? ` (now ${profile.home.name})` : ""}
        </label>
        <div className="field-label">Moods you enjoy</div>
        <div className="chips wrap">{moods.map((m) => (
          <button type="button" key={m} className={"chip" + (picked.includes(m) ? " on" : "")} onClick={() => toggle(m)}>{m}</button>
        ))}</div>
        {err && <small className="danger">Could not save ({err}).</small>}
        <div className="compose-foot">
          {onClose && <button type="button" className="btn ghost" onClick={onClose}>Cancel</button>}
          <button type="submit" className="btn" disabled={!name.trim()}>Save</button>
        </div>
      </form>
    </div>
  );
}
