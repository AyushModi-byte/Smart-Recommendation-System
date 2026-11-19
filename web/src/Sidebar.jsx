// Left sidebar: navigation between Discover and the user's places, with the account block at the bottom.
import { useEffect, useRef, useState } from "react";
import { accountsEnabled } from "./account.js";
import { initial } from "./AccountViews.jsx";
import { Bookmark, Chat, Close, Compass, Edit, Heart, Logout, Pin, Undo, Users } from "./icons.jsx";

export default function Sidebar({ view, setView, counts, user, profile, open, onClose, onSignIn, onSignOut, onEdit, onStartOver }) {
  const [menu, setMenu] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!menu) return;
    const f = (e) => { if (!ref.current?.contains(e.target)) setMenu(false); };
    document.addEventListener("mousedown", f);
    return () => document.removeEventListener("mousedown", f);
  }, [menu]);

  const item = (id, label, Icon, count) => (
    <button key={id} className={"nav-item" + (view === id ? " on" : "")} aria-current={view === id ? "page" : undefined}
            onClick={() => { setView(id); onClose(); }}>
      <Icon size={19} /><span>{label}</span>{count !== undefined && <small>{count}</small>}
    </button>
  );

  return (
    <>
      {open && <div className="scrim" onClick={onClose} />}
      <nav className={"sidebar" + (open ? " open" : "")} aria-label="Main">
        <div className="side-top">
          <div className="brand"><span className="logo"><Pin size={18} /></span><span className="brand-name">AuraPath</span></div>
          <button className="icon-btn only-narrow" onClick={onClose} aria-label="Close menu"><Close size={18} /></button>
        </div>
        <div className="nav">
          {item("discover", "Discover", Compass)}
          {item("liked", "Liked places", Heart, counts.liked)}
          {item("saved", "Saved for later", Bookmark, counts.saved)}
          {item("posts", "My posts", Chat, counts.posts)}
          {accountsEnabled && <div className="nav-label">Community</div>}
          {accountsEnabled && item("people", "People like you", Users)}
          {accountsEnabled && item("community", "Recent posts", Chat)}
        </div>

        <div className="side-user" ref={ref}>
          {menu && (
            <div className="user-menu" role="menu">
              {user && <button role="menuitem" onClick={() => { setMenu(false); setView("profile"); onClose(); }}><Users size={16} /> View profile</button>}
              {user && <button role="menuitem" onClick={() => { setMenu(false); onEdit(); }}><Edit size={16} /> Edit profile</button>}
              <button role="menuitem" onClick={() => { setMenu(false); onStartOver(); }}><Undo size={16} /> Start over on this device</button>
              {user
                ? <button role="menuitem" className="danger" onClick={() => { setMenu(false); onSignOut(); }}><Logout size={16} /> Sign out</button>
                : accountsEnabled && <button role="menuitem" onClick={() => { setMenu(false); onSignIn(); }}><Logout size={16} /> Sign in or create account</button>}
            </div>
          )}
          <button className="user-chip" onClick={() => setMenu((m) => !m)} aria-haspopup="menu" aria-expanded={menu}>
            <span className={"avatar" + (user ? "" : " guest")}>{user ? initial(profile, user) : "G"}</span>
            <span className="user-text">
              <b>{user ? profile?.name || "New user" : "Guest"}</b>
              <small>{user ? user.email : accountsEnabled ? "Not signed in" : "Saved in this browser"}</small>
            </span>
            <span className="dots" aria-hidden="true">⋯</span>
          </button>
        </div>
      </nav>
    </>
  );
}
