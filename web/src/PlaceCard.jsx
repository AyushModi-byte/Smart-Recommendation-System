import { useEffect, useRef, useState } from "react";
import { Pin, Spark, Expand, Camera, Calendar } from "./icons.jsx";

const fmt = (n) => n.toLocaleString("en");

// Several photos per place: arrows / dots / swipe to browse. A photo that fails to load (deleted on Flickr) is
// dropped from the list silently; the placeholder shows only when none of them load.
function Photos({ card, big }) {
  const all = card.photos?.length ? card.photos : card.photo ? [{ src: card.photo, page: card.photo_page }] : [];
  const [broken, setBroken] = useState(() => new Set());
  const [i, setI] = useState(0);
  const touch = useRef(null);
  useEffect(() => { setI(0); setBroken(new Set()); }, [card.id]);
  const list = all.filter((p) => !broken.has(p.src));
  const n = list.length;
  const at = n ? Math.min(i, n - 1) : 0;
  const cls = "photo" + (big ? " big" : "");
  if (!n) return <div className={cls + " photo-empty"}><Camera size={36} /><span>No photo available</span></div>;
  const go = (d) => setI((at + d + n) % n);
  return (
    <div className={cls}
         onTouchStart={(e) => { touch.current = e.touches[0].clientX; }}
         onTouchEnd={(e) => {
           const dx = e.changedTouches[0].clientX - (touch.current ?? 0);
           if (Math.abs(dx) > 40) go(dx < 0 ? 1 : -1);
         }}>
      <a href={list[at].page} target="_blank" rel="noopener noreferrer" title="Open the original photo on Flickr">
        <img key={list[at].src} src={list[at].src} alt={`${card.name}, photo ${at + 1} of ${n}`} loading="eager"
             onError={() => setBroken((b) => new Set(b).add(list[at].src))} />
      </a>
      {/* load the next photo early so browsing is instant, and find dead links before they are shown */}
      {n > 1 && <img className="preload" src={list[(at + 1) % n].src} alt="" aria-hidden="true"
                     onError={() => setBroken((b) => new Set(b).add(list[(at + 1) % n].src))} />}
      {n > 1 && (
        <>
          <button className="nav prev" onClick={() => go(-1)} aria-label="Previous photo"><Chevron dir="left" /></button>
          <button className="nav next" onClick={() => go(1)} aria-label="Next photo"><Chevron dir="right" /></button>
          <div className="dots">{list.map((p, k) => <span key={p.src} className={k === at ? "on" : ""} />)}</div>
        </>
      )}
    </div>
  );
}

const Chevron = ({ dir }) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"
       strokeLinejoin="round" aria-hidden="true">{dir === "left" ? <path d="M15 5l-7 7 7 7" /> : <path d="M9 5l7 7-7 7" />}</svg>
);

export default function PlaceCard({ card, expanded, onToggle }) {
  return (
    <article className={"card" + (expanded ? " expanded" : "")} key={card.id}>
      <div className="photo-box">
        <Photos card={card} big={expanded} />
        <div className="badges">
          <span className="badge">{card.category}</span>
          <span className="badge badge-ml" title="Where this place ranks for you among all places in this area">
            <Spark size={13} /> Top {card.top_pct}%
          </span>
        </div>
        <button className="icon-btn corner" onClick={onToggle} aria-label={expanded ? "Show less" : "Show details"}><Expand size={16} /></button>
      </div>
      <div className="card-body">
        <div className="title-row">
          <h2>{card.name}</h2>
          {card.cc && <span className="cc">{card.cc}</span>}
        </div>
        <div className="where"><Pin size={15} /> {card.town}{card.region && card.region !== card.town ? `, ${card.region}` : ""}, {card.country}</div>
        <p className="desc">{card.description}</p>
        {card.tags.length > 0 && <div className="tags">{card.tags.map((t) => <span key={t} className="tag">{t}</span>)}</div>}
        <div className="reason">{card.reason}</div>
        {expanded && (
          <div className="more">
            <div><b>Mood</b> {card.mood}</div>
            <div><b>Distance</b> {card.distance_km} km from the area centre</div>
            <div><b>Photos</b> {fmt(card.n_photos)} by {fmt(card.n_visitors)} people</div>
            <div className="links">
              <a href={`https://www.google.com/maps/search/?api=1&query=${card.lat},${card.lon}`} target="_blank" rel="noopener noreferrer">Open in Google Maps</a>
              {card.photo_page && <a href={card.photo_page} target="_blank" rel="noopener noreferrer">Photo on Flickr</a>}
            </div>
          </div>
        )}
        <div className="card-foot">
          <span className="stat"><Camera size={15} /> {fmt(card.n_visitors)} {card.n_visitors === 1 ? "visitor" : "visitors"}</span>
          {card.best_months && <span className="season"><Calendar size={14} /> {card.best_months}</span>}
        </div>
      </div>
    </article>
  );
}
