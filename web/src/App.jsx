import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import MapView from "./MapView.jsx";
import PlaceCard from "./PlaceCard.jsx";
import { getMeta, getPlace, recommend, searchTowns, sendFeedback } from "./api.js";
import { Bookmark, Close, Heart, Pin, Search, Target, Undo, X } from "./icons.jsx";

const STORE = "aurapath.v1";
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
  const [filters, setFilters] = useState(saved0.filters || { category: null, mood: null, hidden: false });
  const [session] = useState(saved0.session || newSession());
  const [history, setHistory] = useState([]);                         // undo stack: {card, action}
  const [queue, setQueue] = useState([]);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [compose, setCompose] = useState(null);                       // card being posted about
  const narrow = useIsNarrow();
  const req = useRef(0);

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
      }, setWaking);
      if (id !== req.current) return;
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

  useEffect(() => {
    if (meta) refill(snap.current, null);
  }, [meta, area, filters, refill]);

  const current = queue[0] || null;
  useEffect(() => { if (current) sendFeedback(session, current.id, "shown"); }, [current, session]);

  const act = useCallback((action) => {
    const card = queue[0];
    if (!card) return;
    sendFeedback(session, card.id, action);
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
  }, [queue, session, refill]);

  const undo = useCallback(() => {
    const last = history[history.length - 1];
    if (!last) return;
    const { card, action } = last;
    req.current++;                                   // drop a refill still in flight so it cannot hide the restored card
    setLoading(false);
    sendFeedback(session, card.id, "undo");
    setHistory((h) => h.slice(0, -1));
    setSeen((s) => s.filter((i) => i !== card.id));
    if (action === "accept") {
      setAccepted((a) => a.filter((i) => i !== card.id));
      setLiked((l) => l.filter((c) => c.id !== card.id));
    }
    if (action === "reject") setRejected((r) => r.filter((i) => i !== card.id));
    if (action === "save") setSaved((s) => s.filter((c) => c.id !== card.id));
    setQueue((q) => [card, ...q.filter((c) => c.id !== card.id)]);
  }, [history, session]);

  useEffect(() => {
    const onKey = (e) => {
      if (compose || e.target.closest("input, textarea")) return;
      if (e.key === "ArrowLeft" || e.key === "x") act("reject");
      else if (e.key === "ArrowRight" || e.key === "l") act("accept");
      else if (e.key === "s") act("save");
      else if (e.key === "u" || (e.key === "z" && (e.ctrlKey || e.metaKey))) undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [act, undo, compose]);

  const setAreaTo = (a) => { setQueue([]); setArea({ ...area, ...a }); };
  const startOver = () => {
    if (!window.confirm("Forget your likes, skips and saved places on this device?")) return;
    setAccepted([]); setRejected([]); setSeen([]); setLiked([]); setSaved([]); setPosts([]); setHistory([]); setQueue([]);
    setDrawer(false);
    refill({ ...snap.current, accepted: [], rejected: [], seen: [] }, null);
  };

  return (
    <div className="app">
      <MapView area={area} current={current} upcoming={queue.slice(1)} liked={liked}
               onSearchArea={(lat, lon) => setAreaTo({ name: "This area", lat, lon })} padRight={narrow ? 0 : 460} />

      <header className="topbar">
        <div className="brand">
          <span className="logo"><Pin size={18} /></span>
          <span className="brand-name">AuraPath</span>
        </div>
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
          <select className={"chip select" + (filters.mood ? " on" : "")} value={filters.mood || ""} aria-label="Mood"
                  onChange={(e) => setFilters({ ...filters, mood: e.target.value || null })}>
            <option value="">Any mood</option>
            {(meta?.moods || []).map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <button className="counter" onClick={() => setDrawer(true)}>
          <span><Bookmark size={14} /> {saved.length}</span>
          <span><Heart size={14} /> {liked.length}</span>
        </button>
      </header>

      <aside className="panel">
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

      {drawer && (
        <Drawer liked={liked} saved={saved} posts={posts} onClose={() => setDrawer(false)} onStartOver={startOver}
                onShow={(c) => { setDrawer(false); setQueue((q) => [c, ...q.filter((x) => x.id !== c.id)]); }} />
      )}
      {compose && (
        <Compose card={compose} onClose={() => setCompose(null)}
                 onPost={(text) => { setPosts((p) => [{ id: Date.now(), place: compose, text, time: new Date().toISOString() }, ...p]); setCompose(null); }} />
      )}
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

function Drawer({ liked, saved, posts, onClose, onShow, onStartOver }) {
  const [tab, setTab] = useState("liked");
  const items = tab === "liked" ? liked : saved;
  return (
    <div className="overlay" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Your places">
        <div className="drawer-head">
          <div className="tabs">
            <button className={tab === "liked" ? "on" : ""} onClick={() => setTab("liked")}>Liked {liked.length}</button>
            <button className={tab === "saved" ? "on" : ""} onClick={() => setTab("saved")}>Saved {saved.length}</button>
            <button className={tab === "posts" ? "on" : ""} onClick={() => setTab("posts")}>Posts {posts.length}</button>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><Close size={18} /></button>
        </div>
        <div className="drawer-body">
          {tab === "posts" ? (
            posts.length ? posts.map((p) => (
              <div key={p.id} className="post">
                <div className="post-place">{p.place.name} <small>{p.place.town}</small></div>
                <p>{p.text}</p>
                <small>{new Date(p.time).toLocaleString()}</small>
              </div>
            )) : <p className="muted">When you like a place you can post about it. Posts stay on this device for now.</p>
          ) : items.length ? items.map((c) => (
            <button key={c.id} className="mini" onClick={() => onShow(c)}>
              <img src={c.photo} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
              <span><b>{c.name}</b><small>{c.town}, {c.country} · {c.category}</small></span>
            </button>
          )) : <p className="muted">Nothing here yet.</p>}
        </div>
        <button className="link danger" onClick={onStartOver}>Start over</button>
      </div>
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
