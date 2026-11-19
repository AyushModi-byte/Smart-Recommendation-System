import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import MapView from "./MapView.jsx";
import PlaceCard from "./PlaceCard.jsx";
import { getMeta, getPlace, getTaste, recommend, searchTowns, sendFeedback } from "./api.js";
import * as acct from "./account.js";
import { Community, Login, NewPassword, Onboarding, People, PlaceList, Profile } from "./AccountViews.jsx";
import Sidebar from "./Sidebar.jsx";
import { Bookmark, Close, Heart, Menu, Pin, Search, Target, Undo, X } from "./icons.jsx";

const STORE = "aurapath.v1";
const GUEST = "aurapath.guest";
const DEFAULT_AREA = { name: "London", lat: 51.5074, lon: -0.1278, radius: 15 };

function load() {
  try {
    return JSON.parse(localStorage.getItem(STORE)) || {};
  } catch {
    return {};
  }
}
function newSession() {
  try {
    return crypto.randomUUID();
  } catch {
    return String(Math.random()).slice(2);
  }
}

function useIsNarrow() {
  const q = "(max-width: 760px)";
  const [narrow, setNarrow] = useState(() => window.matchMedia(q).matches);
  useEffect(() => {
    const m = window.matchMedia(q);
    const f = () => setNarrow(m.matches);
    m.addEventListener("change", f);
    return () => m.removeEventListener("change", f);
  }, []);
  return narrow;
}

export default function App() {
  const saved0 = useMemo(load, []);
  const [area, setArea] = useState(saved0.area || DEFAULT_AREA);
  const [accepted, setAccepted] = useState(saved0.accepted || []);   // ids, oldest first
  const [rejected, setRejected] = useState(saved0.rejected || []);
  const [seen, setSeen] = useState(saved0.seen || []);
  const [liked, setLiked] = useState(saved0.liked || []);            // cards
  const [saved, setSaved] = useState(saved0.saved || []);            // cards
  const [posts, setPosts] = useState(saved0.posts || []);
  const [filters, setFilters] = useState({ category: null, mood: null, hidden: false, weather: true, quiet: false, ...(saved0.filters || {}) });
  const [live, setLive] = useState(null);                              // {conditions, notes} from the last batch
  const [session] = useState(saved0.session || newSession());
  const [history, setHistory] = useState([]);                         // undo stack: {card, action}
  const [queue, setQueue] = useState([]);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [compose, setCompose] = useState(null);                       // card being posted about
  const narrow = useIsNarrow();
  const req = useRef(0);
  const [user, setUser] = useState(null);                              // Supabase user, when accounts are on
  const [profile, setProfile] = useState(null);
  const [authReady, setAuthReady] = useState(!acct.sb);
  const [guest, setGuest] = useState(() => { try { return localStorage.getItem(GUEST) === "1"; } catch { return false; } });
  const [recovery, setRecovery] = useState(false);
  const [onboard, setOnboard] = useState(false);
  const [view, setView] = useState("discover");                       // discover | liked | saved | posts | people | community | profile
  const [navOpen, setNavOpen] = useState(false);
  const [myTaste, setMyTaste] = useState(null);
  const [tasteErr, setTasteErr] = useState("");
  const synced = useRef(null);                                        // user id whose data has been merged

  useEffect(() => {
    if (!acct.sb) return;
    acct.sb.auth.getSession().then(({ data }) => { setUser(data.session?.user || null); setAuthReady(true); });
    const { data } = acct.sb.auth.onAuthStateChange((ev, s) => {
      setUser(s?.user || null);
      if (ev === "PASSWORD_RECOVERY") setRecovery(true);
    });
    return () => data.subscription.unsubscribe();
  }, []);

  // signed in: load the profile (or ask for one)
  useEffect(() => {
    if (!user) { setProfile(null); synced.current = null; return; }
    acct.loadProfile(user.id).then((p) => { if (p) setProfile(p); else setOnboard(true); })
      .catch((e) => setError("Could not load your profile. " + e.message));
  }, [user]);

  useEffect(() => {
    try {
      localStorage.setItem(STORE, JSON.stringify({ area, accepted, rejected, seen, liked, saved, posts, filters, session }));
    } catch { /* private mode: state just is not kept */ }
  }, [area, accepted, rejected, seen, liked, saved, posts, filters, session]);

  useEffect(() => {
    getMeta(setWaking).then(setMeta).catch((e) => setError("Could not reach the server. " + e.message));
  }, []);

  // Liked / saved places are stored as card copies; after a data update (new photos, names) those copies are stale.
  // Refresh them from the server once per load, keeping the fields only a recommendation has (reason, rank).
  useEffect(() => {
    if (!meta) return;
    const ids = [...new Set([...liked, ...saved, ...posts.map((p) => p.place)].map((c) => c.id))].slice(0, 300);
    Promise.all(ids.map((id) => getPlace(id).then((c) => [id, c]).catch(() => null))).then((rows) => {
      const fresh = new Map(rows.filter(Boolean));
      const upd = (c) => (fresh.has(c.id) ? { ...c, ...fresh.get(c.id) } : c);
      setLiked((l) => l.map(upd));
      setSaved((s) => s.map(upd));
      setPosts((p) => p.map((x) => ({ ...x, place: upd(x.place) })));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta]);

  // ask for a fresh batch; keep the card on screen (if any) at the front
  const refill = useCallback(async (state, keepFirst) => {
    const id = ++req.current;
    setLoading(true);
    setError("");
    try {
      const r = await recommend({
        lat: state.area.lat, lon: state.area.lon, radius_km: state.area.radius,
        accepted: state.accepted.slice(-200), rejected: state.rejected.slice(-200),
        seen: [...state.seen, ...(keepFirst ? [keepFirst.id] : [])].slice(-600),
        mood: state.filters.mood, category: state.filters.category, hidden: state.filters.hidden, n: 6,
        weather: state.filters.weather, quiet: state.filters.quiet,
      }, setWaking);
      if (id !== req.current) return;
      setLive({ conditions: r.conditions || null, notes: r.notes || [] });
      setQueue(keepFirst ? [keepFirst, ...r.cards.filter((c) => c.id !== keepFirst.id)] : r.cards);
      if (!r.cards.length && !keepFirst) setError("Nothing left to show here. Widen the area, clear a filter, or search another town.");
    } catch (e) {
      if (id === req.current) setError("Could not load places. " + e.message);
    } finally {
      if (id === req.current) setLoading(false);
    }
  }, []);

  const snapshot = { area, accepted, rejected, seen, filters };
  const snap = useRef(snapshot);
  snap.current = snapshot;

  // signed in with a profile: merge this device's likes, saves, skips and posts with the account, once per user
  useEffect(() => {
    if (!user || !profile || synced.current === user.id) return;
    synced.current = user.id;
    if (profile.home) setArea((a) => ({ ...a, ...profile.home }));
    (async () => {
      const st = snap.current;
      const srv = await acct.loadActions();
      const has = { accept: new Set(srv.accept.map((c) => c.id)), reject: new Set(srv.reject), save: new Set(srv.save.map((c) => c.id)) };
      const likedById = new Map(liked.map((c) => [c.id, c]));
      const savedIds = new Set(saved.map((c) => c.id));
      await acct.putActions([
        ...st.accepted.filter((i) => !has.accept.has(i))
          .map((i) => ({ place_id: i, action: "accept", card: likedById.has(i) ? acct.slimCard(likedById.get(i)) : null })),
        ...st.rejected.filter((i) => !has.reject.has(i)).map((i) => ({ place_id: i, action: "reject", card: null })),
        ...saved.filter((c) => !has.save.has(c.id)).map((c) => ({ place_id: c.id, action: "save", card: acct.slimCard(c) })),
      ]);
      const acc = [...srv.accept.map((c) => c.id), ...st.accepted.filter((i) => !has.accept.has(i))];
      const rej = [...srv.reject, ...st.rejected.filter((i) => !has.reject.has(i))];
      setAccepted(acc);
      setRejected(rej);
      setSeen((s) => [...new Set([...s, ...acc, ...rej])]);
      setLiked((l) => [...l, ...srv.accept.filter((c) => !likedById.has(c.id)).reverse()]);
      setSaved((v) => [...v, ...srv.save.filter((c) => !savedIds.has(c.id)).reverse()]);
      for (const p of posts.filter((x) => !x.synced)) await acct.addPost(p.place, p.text);
      const mine = await acct.myPosts(user.id);
      setPosts(mine.map((x) => ({ id: x.id, place: x.place, text: x.body, time: x.created_at, synced: true })));
    })().catch((e) => setError("Could not sync your account. " + e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, profile]);

  const online = !!(user && profile);
  const pushAction = (card, action) => {
    if (online) acct.putActions([{ place_id: card.id, action, card: action === "reject" ? null : acct.slimCard(card) }]).catch(() => {});
  };
  const dropAction = (card, action) => { if (online) acct.deleteAction(card.id, action).catch(() => {}); };

  useEffect(() => { try { localStorage.setItem(GUEST, guest ? "1" : "0"); } catch { /* ignore */ } }, [guest]);

  // signed in: keep the shared taste row (for "people like you") in step with the likes, a moment after they change
  useEffect(() => {
    if (!online || accepted.length < 3) { setMyTaste(null); setTasteErr(""); return; }
    const t = setTimeout(async () => {
      let step = "work out your taste";
      try {
        const taste = await getTaste(accepted.slice(-200));
        setTasteErr(taste ? "" : "Could not work out your taste from these places.");
        setMyTaste(taste);
        if (!taste) return;
        step = "share your taste";
        if (profile.discoverable === false) { await acct.deleteTaste(user.id); return; }
        const counts = {};
        for (const c of liked) if (c.category) counts[c.category] = (counts[c.category] || 0) + 1;
        await acct.putTaste({
          user_id: user.id, taste, favorites: liked.slice(0, 8).map(acct.slimCard), liked_count: liked.length,
          top_categories: Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3).map((x) => x[0]),
          updated_at: new Date().toISOString(),
        });
      } catch (e) { setTasteErr(`Could not ${step}: ${e.message || e}`); }
    }, 1500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [online, accepted, profile?.discoverable]);

  useEffect(() => {
    if (meta) refill(snap.current, null);
  }, [meta, area, filters, refill]);

  const current = queue[0] || null;
  useEffect(() => { if (current) sendFeedback(session, current.id, "shown"); }, [current, session]);

  const act = useCallback((action) => {
    const card = queue[0];
    if (!card) return;
    sendFeedback(session, card.id, action);
    pushAction(card, action);
    const next = { ...snap.current, seen: [...snap.current.seen, card.id] };
    if (action === "accept") {
      next.accepted = [...snap.current.accepted, card.id];
      setAccepted(next.accepted);
      setLiked((l) => [card, ...l.filter((c) => c.id !== card.id)]);
      setCompose(card);
    } else if (action === "reject") {
      next.rejected = [...snap.current.rejected, card.id];
      setRejected(next.rejected);
    } else if (action === "save") {
      setSaved((s) => [card, ...s.filter((c) => c.id !== card.id)]);
    }
    setSeen(next.seen);
    setHistory((h) => [...h.slice(-30), { card, action }]);
    setExpanded(false);
    const rest = queue.slice(1);
    setQueue(rest);
    refill(next, rest[0] || null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queue, session, refill, online]);

  const undo = useCallback(() => {
    const last = history[history.length - 1];
    if (!last) return;
    const { card, action } = last;
    req.current++;                                   // drop a refill still in flight so it cannot hide the restored card
    setLoading(false);
    sendFeedback(session, card.id, "undo");
    dropAction(card, action);
    setHistory((h) => h.slice(0, -1));
    setSeen((s) => s.filter((i) => i !== card.id));
    if (action === "accept") {
      setAccepted((a) => a.filter((i) => i !== card.id));
      setLiked((l) => l.filter((c) => c.id !== card.id));
    }
    if (action === "reject") setRejected((r) => r.filter((i) => i !== card.id));
    if (action === "save") setSaved((s) => s.filter((c) => c.id !== card.id));
    setQueue((q) => [card, ...q.filter((c) => c.id !== card.id)]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history, session, online]);

  useEffect(() => {
    const onKey = (e) => {
      if (compose || onboard || recovery || e.target.closest("input, textarea, select")) return;
      if (e.key === "ArrowLeft" || e.key === "x") act("reject");
      else if (e.key === "ArrowRight" || e.key === "l") act("accept");
      else if (e.key === "s") act("save");
      else if (e.key === "u" || (e.key === "z" && (e.ctrlKey || e.metaKey))) undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [act, undo, compose, onboard, recovery]);

  const setAreaTo = (a) => { setQueue([]); setArea({ ...area, ...a }); };
  const startOver = () => {
    if (!window.confirm("Forget your likes, skips and saved places on this device?")) return;
    clearLocal();
    refill({ ...snap.current, accepted: [], rejected: [], seen: [] }, null);
  };
  const clearLocal = () => {
    setAccepted([]); setRejected([]); setSeen([]); setLiked([]); setSaved([]); setPosts([]); setHistory([]); setQueue([]);
  };
  // signing out also clears this device, so the next person to sign in here does not inherit these likes
  const doSignOut = async () => { await acct.signOut(); clearLocal(); setGuest(false); setView("discover"); };
  const showCard = (c) => { if (narrow) setView("discover"); setQueue((q) => [c, ...q.filter((x) => x.id !== c.id)]); };

  if (acct.accountsEnabled && !authReady) {
    return <div className="splash"><span className="logo big"><Pin size={26} /></span></div>;
  }
  if (acct.accountsEnabled && !user && !guest) return <Login onGuest={() => setGuest(true)} />;

  const sheetOpen = view !== "discover";
  const TITLES = { liked: "Liked places", saved: "Saved for later", posts: "My posts", people: "People like you",
                   community: "Recent posts", profile: "Your profile" };

  return (
    <div className="app">
      <Sidebar view={view} setView={setView} counts={{ liked: liked.length, saved: saved.length, posts: posts.length }}
               user={user} profile={profile} open={navOpen} onClose={() => setNavOpen(false)}
               onSignIn={() => setGuest(false)} onSignOut={doSignOut} onEdit={() => setOnboard(true)} onStartOver={startOver} />

      <main className="stage">
        <MapView area={area} current={current} upcoming={queue.slice(1)} liked={liked}
                 onSearchArea={(lat, lon) => setAreaTo({ name: "This area", lat, lon })}
                 padRight={narrow ? 0 : 460} padLeft={!narrow && sheetOpen ? 400 : 0} />

        <header className="topbar">
          <button className="icon-btn only-narrow" onClick={() => setNavOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
          <TownSearch area={area} towns={meta?.towns || []} onPick={setAreaTo} />
          <div className="chips" role="toolbar" aria-label="Filters">
            <button className={"chip" + (filters.category === null && !filters.hidden ? " on" : "")}
                    onClick={() => setFilters({ ...filters, category: null, hidden: false })}>All</button>
            {(meta?.categories || []).map((c) => (
              <button key={c.id} className={"chip" + (filters.category === c.id ? " on" : "")}
                      onClick={() => setFilters({ ...filters, category: filters.category === c.id ? null : c.id })}>{c.name}</button>
            ))}
            <button className={"chip" + (filters.hidden ? " on" : "")}
                    onClick={() => setFilters({ ...filters, hidden: !filters.hidden })}>Hidden gems</button>
            <button className={"chip" + (filters.quiet ? " on" : "")} title="Favour places that are usually quiet at this time of day"
                    onClick={() => setFilters({ ...filters, quiet: !filters.quiet })}>Quieter now</button>
            <button className={"chip" + (filters.weather ? " on" : "")} title="Reorder for rain, cold or heat in this area right now"
                    onClick={() => setFilters({ ...filters, weather: !filters.weather })}>Weather-aware</button>
            <select className={"chip select" + (filters.mood ? " on" : "")} value={filters.mood || ""} aria-label="Mood"
                    onChange={(e) => setFilters({ ...filters, mood: e.target.value || null })}>
              <option value="">Any mood</option>
              {(meta?.moods || []).map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
        </header>

        {sheetOpen && (
          <section className="sheet" aria-label={TITLES[view]}>
            <div className="sheet-head">
              <h2>{TITLES[view]}</h2>
              <button className="icon-btn" onClick={() => setView("discover")} aria-label="Close"><Close size={18} /></button>
            </div>
            <div className="sheet-body">
              {view === "liked" && <PlaceList items={liked} onShow={showCard} empty="Places you like show up here. Tap the heart on a card." />}
              {view === "saved" && <PlaceList items={saved} onShow={showCard} empty="Save places for later with the bookmark button." />}
              {view === "posts" && (posts.length ? posts.map((p) => (
                <div key={p.id} className="post">
                  <div className="post-place">{p.place.name} <small>{p.place.town}</small></div>
                  <p>{p.text}</p>
                  <small>{new Date(p.time).toLocaleString()}</small>
                </div>
              )) : <p className="muted">When you like a place you can post about it.{user ? "" : acct.accountsEnabled ? " Sign in to share posts." : " Posts stay on this device."}</p>)}
              {view === "community" && <Community />}
              {view === "people" && (user
                ? <People user={user} taste={myTaste} tasteErr={tasteErr}likedCount={accepted.length} likedIds={new Set(accepted)} onShow={showCard} />
                : <p className="muted">Sign in to see people whose taste matches yours.</p>)}
              {view === "profile" && user && (
                <Profile user={user} profile={profile} liked={liked} posts={posts}
                         onMood={(m) => { setView("discover"); setFilters({ ...filters, mood: m }); }}
                         onHome={() => { setView("discover"); if (profile?.home) setAreaTo(profile.home); }}
                         onEdit={() => setOnboard(true)} />
              )}
            </div>
          </section>
        )}

        <aside className={"panel" + (narrow && sheetOpen ? " hidden" : "")}>
          <LiveLine live={live} />
          {current ? (
            <PlaceCard card={current} expanded={expanded} onToggle={() => setExpanded((x) => !x)} />
          ) : (
            <div className="card placeholder">
              {error ? <p>{error}</p> : waking ? <p>Waking the server up. Free hosting sleeps when idle; this can take up to a minute.</p>
                : <p>{loading ? "Finding places…" : "Pick a town to start."}</p>}
            </div>
          )}
          {current && error && <div className="toast">{error}</div>}
          <div className="actions">
            <button className="act small" onClick={undo} disabled={!history.length} aria-label="Undo" title="Undo (U)"><Undo size={18} /></button>
            <button className="act" onClick={() => act("reject")} disabled={!current} aria-label="Skip" title="Skip (←)"><X size={22} /></button>
            <button className="act gold" onClick={() => act("save")} disabled={!current} aria-label="Save for later" title="Save (S)"><Bookmark size={20} /></button>
            <button className="act love" onClick={() => act("accept")} disabled={!current} aria-label="I'd go here" title="Like (→)"><Heart size={26} /></button>
          </div>
          <div className="hint">{accepted.length ? `Learning from ${accepted.length} liked ${accepted.length === 1 ? "place" : "places"}` : "Like a few places and the feed adapts to your taste"}</div>
        </aside>
      </main>

      {compose && (
        <Compose card={compose} onClose={() => setCompose(null)}
                 onPost={(text) => {
                   const card = compose;
                   setCompose(null);
                   if (online) {
                     acct.addPost(card, text)
                       .then((x) => setPosts((p) => [{ id: x.id, place: x.place, text: x.body, time: x.created_at, synced: true }, ...p]))
                       .catch((e) => setError("Could not post. " + e.message));
                   } else {
                     setPosts((p) => [{ id: Date.now(), place: card, text, time: new Date().toISOString() }, ...p]);
                   }
                 }} />
      )}
      {onboard && user && (
        <Onboarding profile={profile} area={area} moods={meta?.moods || []} onClose={profile ? () => setOnboard(false) : null}
                    onSave={async (p) => { const row = await acct.saveProfile(user.id, p); setProfile(row); setOnboard(false); }} />
      )}
      {recovery && <NewPassword onDone={() => setRecovery(false)} />}
    </div>
  );
}

function LiveLine({ live }) {
  const c = live?.conditions;
  if (!c && !live?.notes?.length) return null;
  return (
    <div className="live" aria-live="polite">
      {c && <span className="live-now">{c.rain ? "Rain" : c.is_day ? "Dry" : "Dry, dark"} · {c.temp_c} °C · {c.local_time} local</span>}
      {live.notes.map((n) => <span key={n} className="live-note">{n}</span>)}
    </div>
  );
}

function TownSearch({ area, towns, onPick }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [hits, setHits] = useState([]);
  useEffect(() => {
    if (q.trim().length < 2) { setHits([]); return; }
    const t = setTimeout(() => searchTowns(q).then(setHits).catch(() => setHits([])), 200);
    return () => clearTimeout(t);
  }, [q]);
  const list = q.trim().length >= 2 ? hits : towns;
  const pick = (t) => { onPick({ name: t.name, lat: t.lat, lon: t.lon }); setQ(""); setOpen(false); };
  const locate = () => navigator.geolocation?.getCurrentPosition(
    (p) => { onPick({ name: "Near you", lat: p.coords.latitude, lon: p.coords.longitude }); setOpen(false); },
    () => alert("Location permission was denied."));
  return (
    <div className="town" onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setOpen(false); }}>
      <Search size={16} />
      <input value={q} placeholder={area.name} onFocus={() => setOpen(true)} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
             onKeyDown={(e) => { if (e.key === "Enter" && list[0]) pick(list[0]); }} aria-label="Search a town" />
      {open && (
        <div className="town-list">
          <button className="town-item locate" onMouseDown={(e) => e.preventDefault()} onClick={locate}><Target size={15} /> Use my location</button>
          {list.map((t) => (
            <button key={`${t.name}-${t.lat}`} className="town-item" onMouseDown={(e) => e.preventDefault()} onClick={() => pick(t)}>
              <span>{t.name}</span><small>{[t.region, t.country].filter(Boolean).join(", ")} · {t.n_places.toLocaleString("en")} places</small>
            </button>
          ))}
          {q.trim().length >= 2 && !list.length && <div className="town-empty">No town with photo data matches “{q}”.</div>}
          <div className="radius">
            Area
            {[5, 15, 30, 60].map((r) => (
              <button key={r} className={"chip tiny" + (area.radius === r ? " on" : "")} onMouseDown={(e) => e.preventDefault()}
                      onClick={() => onPick({ radius: r })}>{r} km</button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Compose({ card, onClose, onPost }) {
  const [text, setText] = useState("");
  return (
    <div className="overlay" onClick={onClose}>
      <form className="compose" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Post about this place"
            onSubmit={(e) => { e.preventDefault(); if (text.trim()) onPost(text.trim()); }}>
        <div className="compose-head"><Heart size={16} /> You liked <b>{card.name}</b></div>
        <textarea autoFocus maxLength={280} rows={4} value={text} onChange={(e) => setText(e.target.value)}
                  placeholder="Say something about it: been there, want to go, a tip…" />
        <div className="compose-foot">
          <small>{280 - text.length}</small>
          <button type="button" className="btn ghost" onClick={onClose}>Skip</button>
          <button type="submit" className="btn" disabled={!text.trim()}>Post</button>
        </div>
      </form>
    </div>
  );
}
