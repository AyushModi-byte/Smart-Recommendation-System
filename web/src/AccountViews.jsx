// Account screens: login page, password reset, first-time profile, profile, community feed, people with similar taste.
import { useEffect, useState } from "react";
import * as acct from "./account.js";
import { Pin } from "./icons.jsx";

export const initial = (profile, user) => (profile?.name || user?.email || "?").slice(0, 1).toUpperCase();

// ---------------------------------------------------------------- login page
export function Login({ onGuest }) {
  const [mode, setMode] = useState("signin");                       // signin | signup | link | forgot
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);                              // {kind: "ok" | "err", text}

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setMsg(null);
    try {
      if (mode === "signin") await acct.signInPassword(email.trim(), password);
      else if (mode === "signup") {
        const r = await acct.signUpPassword(email.trim(), password);
        if (!r.session) setMsg({ kind: "ok", text: `Almost there: confirm your address from the email we sent to ${email.trim()}.` });
      } else if (mode === "link") {
        await acct.sendMagicLink(email.trim());
        setMsg({ kind: "ok", text: `Check your inbox: a sign-in link is on its way to ${email.trim()}.` });
      } else {
        await acct.resetPassword(email.trim());
        setMsg({ kind: "ok", text: `If an account exists for ${email.trim()}, a reset link is on its way.` });
      }
    } catch (err) {
      setMsg({ kind: "err", text: err.message });
    } finally {
      setBusy(false);
    }
  };
  const go = (m) => { setMode(m); setMsg(null); };
  const needsPw = mode === "signin" || mode === "signup";
  const cta = { signin: "Sign in", signup: "Create account", link: "Email me a link", forgot: "Send reset link" }[mode];

  return (
    <div className="login">
      <section className="login-hero">
        <div className="brand"><span className="logo"><Pin size={18} /></span><span className="brand-name">AuraPath</span></div>
        <div className="hero-copy">
          <h1>Find places that match <em>your</em> taste.</h1>
          <p>Like a few places and AuraPath learns what you enjoy, from the photos millions of travellers took. Then it
            shows you more like them, wherever you are.</p>
          <ul className="hero-points">
            <li><b>177,564</b><span>places across the USA and Europe</span></li>
            <li><b>2.6×</b><span>more accurate than suggesting what is popular nearby</span></li>
            <li><b>Live</b><span>weather and crowd levels shape the feed</span></li>
          </ul>
        </div>
        <small className="hero-foot">A B.Tech project, LNMIIT Jaipur</small>
      </section>

      <section className="login-side">
        <form className="login-card" onSubmit={submit}>
          <h2>{{ signin: "Welcome back", signup: "Create your account", link: "Sign in with a link", forgot: "Reset your password" }[mode]}</h2>
          {(mode === "signin" || mode === "signup") && (
            <div className="seg" role="tablist">
              <button type="button" role="tab" aria-selected={mode === "signin"} className={mode === "signin" ? "on" : ""} onClick={() => go("signin")}>Sign in</button>
              <button type="button" role="tab" aria-selected={mode === "signup"} className={mode === "signup" ? "on" : ""} onClick={() => go("signup")}>Create account</button>
            </div>
          )}
          <label className="field-label">Email
            <input className="field" type="email" required autoFocus autoComplete="email" value={email}
                   onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
          </label>
          {needsPw && (
            <label className="field-label">Password
              <input className="field" type="password" required minLength={6} value={password}
                     autoComplete={mode === "signup" ? "new-password" : "current-password"}
                     onChange={(e) => setPassword(e.target.value)} placeholder={mode === "signup" ? "At least 6 characters" : ""} />
            </label>
          )}
          {mode === "signin" && <button type="button" className="text-link right" onClick={() => go("forgot")}>Forgot password?</button>}
          {msg && <p className={"notice " + msg.kind}>{msg.text}</p>}
          <button type="submit" className="btn wide" disabled={busy}>{busy ? "Please wait…" : cta}</button>
          <div className="or"><span>or</span></div>
          {mode === "link" || mode === "forgot"
            ? <button type="button" className="btn ghost wide" onClick={() => go("signin")}>Back to sign in with password</button>
            : <button type="button" className="btn ghost wide" onClick={() => go("link")}>Email me a sign-in link instead</button>}
          <button type="button" className="text-link center" onClick={onGuest}>Continue without an account</button>
          <small className="fine">Without an account, your likes stay in this browser and posts are not shared.</small>
        </form>
      </section>
    </div>
  );
}

export function NewPassword({ onDone }) {
  const [pw, setPw] = useState("");
  const [err, setErr] = useState("");
  const submit = async (e) => {
    e.preventDefault();
    try { await acct.updatePassword(pw); onDone(); } catch (x) { setErr(x.message); }
  };
  return (
    <div className="overlay center">
      <form className="compose" onSubmit={submit} role="dialog" aria-label="Choose a new password">
        <div className="compose-head">Choose a new password</div>
        <input className="field" type="password" required minLength={6} autoFocus autoComplete="new-password" value={pw}
               onChange={(e) => setPw(e.target.value)} placeholder="At least 6 characters" />
        {err && <small className="danger">{err}</small>}
        <div className="compose-foot"><button type="submit" className="btn">Save password</button></div>
      </form>
    </div>
  );
}

// ---------------------------------------------------------------- first-time / edit profile
export function Onboarding({ profile, area, moods, onSave, onClose }) {
  const [name, setName] = useState(profile?.name || "");
  const [picked, setPicked] = useState(profile?.moods || []);
  const [useArea, setUseArea] = useState(!profile?.home);
  const [discoverable, setDiscoverable] = useState(profile?.discoverable ?? true);
  const [err, setErr] = useState("");
  const toggle = (m) => setPicked((p) => (p.includes(m) ? p.filter((x) => x !== m) : [...p, m]));
  const submit = async (e) => {
    e.preventDefault();
    try {
      await onSave({ name: name.trim(), moods: picked, discoverable,
                     home: useArea ? { name: area.name, lat: area.lat, lon: area.lon, radius: area.radius } : profile?.home || null });
    } catch (x) { setErr(x.message); }
  };
  return (
    <div className="overlay center" onClick={onClose || undefined}>
      <form className="compose" onClick={(e) => e.stopPropagation()} onSubmit={submit} role="dialog" aria-label="Your profile">
        <div className="compose-head">{profile ? "Edit your profile" : "Welcome to AuraPath. Tell us a little about you"}</div>
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
        <label className="check">
          <input type="checkbox" checked={discoverable} onChange={(e) => setDiscoverable(e.target.checked)} />
          Let people with similar taste find me (shows my name, recent likes and scenery types)
        </label>
        {err && <small className="danger">Could not save ({err}).</small>}
        <div className="compose-foot">
          {onClose && <button type="button" className="btn ghost" onClick={onClose}>Cancel</button>}
          <button type="submit" className="btn" disabled={!name.trim()}>Save</button>
        </div>
      </form>
    </div>
  );
}

// ---------------------------------------------------------------- sheet contents
export function PlaceList({ items, onShow, empty }) {
  if (!items.length) return <p className="muted">{empty}</p>;
  return items.map((c) => (
    <button key={c.id} className="mini" onClick={() => onShow(c)}>
      <img src={c.photo} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
      <span><b>{c.name}</b><small>{[c.town, c.country].filter(Boolean).join(", ")}{c.category ? ` · ${c.category}` : ""}</small></span>
    </button>
  ));
}

export function Profile({ user, profile, liked, posts, onMood, onHome, onEdit }) {
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
      {top.length > 0 && (
        <div className="bars">{top.map(([k, n]) => (
          <div key={k} className="bar-row"><span>{k}</span><div><i style={{ width: `${Math.round((100 * n) / liked.length)}%` }} /></div>
            <small>{Math.round((100 * n) / liked.length)}%</small></div>
        ))}</div>
      )}
      {profile?.home && <button className="text-link" onClick={onHome}>Show places around {profile.home.name}</button>}
      {profile?.moods?.length > 0 && (
        <div className="chips wrap">{profile.moods.map((m) => <button key={m} className="chip" onClick={() => onMood(m)}>{m}</button>)}</div>
      )}
      <small className="fine">{profile?.discoverable === false ? "Hidden from people with similar taste." : "Visible to people with similar taste."}</small>
      <button className="btn ghost" onClick={onEdit}>Edit profile</button>
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

export function People({ user, taste, likedCount, likedIds, onShow }) {
  const [rows, setRows] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    if (!taste) return;
    setRows(null);
    acct.similarUsers(user.id, taste).then(setRows).catch((e) => setErr(e.message));
  }, [user.id, taste]);
  if (likedCount < 3) return <p className="muted">Like at least three places, then we’ll find people whose taste matches yours.</p>;
  if (err) return <p className="muted">Could not load people. {err}</p>;
  if (!taste || !rows) return <p className="muted">Finding people like you…</p>;
  if (!rows.length) return <p className="muted">No one else has shared their taste yet. Check back later.</p>;
  return (
    <>
      <p className="muted small">People whose liked places look most like yours. Tap a place to see it.</p>
      {rows.map((r) => {
        const fresh = (r.favorites || []).filter((c) => !likedIds.has(c.id)).slice(0, 4);
        return (
          <div key={r.user_id} className="person">
            <div className="person-head">
              <span className="avatar">{(r.name || "?").slice(0, 1).toUpperCase()}</span>
              <span className="person-name"><b>{r.name}</b><small>{r.liked_count} liked · {(r.top_categories || []).slice(0, 2).join(", ")}</small></span>
              <span className="match" title="Taste match">{r.match}%</span>
            </div>
            {fresh.length > 0 && (
              <div className="thumbs">{fresh.map((c) => (
                <button key={c.id} className="thumb" onClick={() => onShow(c)} title={`${c.name}, ${c.town}`}>
                  <img src={c.photo} alt={c.name} loading="lazy" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
                  <span>{c.name}</span>
                </button>
              ))}</div>
            )}
          </div>
        );
      })}
    </>
  );
}
