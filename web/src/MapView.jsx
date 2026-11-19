import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";

// Free dark basemap (CARTO Dark Matter, OpenStreetMap data); attribution is shown by the map.
const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

function circle(lat, lon, km, n = 96) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI;
    const dLat = (km / 111.32) * Math.cos(a);
    const dLon = (km / (111.32 * Math.cos((lat * Math.PI) / 180))) * Math.sin(a);
    pts.push([lon + dLon, lat + dLat]);
  }
  return { type: "Feature", geometry: { type: "LineString", coordinates: pts } };
}

const fc = (cards) => ({
  type: "FeatureCollection",
  features: cards.map((c) => ({ type: "Feature", properties: { id: c.id, name: c.name },
                                geometry: { type: "Point", coordinates: [c.lon, c.lat] } })),
});

export default function MapView({ area, current, upcoming, liked, onSearchArea, padRight, padLeft = 0 }) {
  const box = useRef(null);
  const map = useRef(null);
  const pin = useRef(null);
  const [ready, setReady] = useState(false);
  const [moved, setMoved] = useState(false);

  useEffect(() => {
    const m = new maplibregl.Map({
      container: box.current, style: STYLE, center: [area.lon, area.lat], zoom: 11.5,
      attributionControl: { compact: true },
    });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-left");
    m.on("load", () => {
      m.addSource("area", { type: "geojson", data: circle(area.lat, area.lon, area.radius) });
      m.addLayer({ id: "area", type: "line", source: "area",
                   paint: { "line-color": "#e8820e", "line-opacity": 0.45, "line-width": 1.5, "line-dasharray": [3, 3] } });
      m.addSource("liked", { type: "geojson", data: fc([]) });
      m.addLayer({ id: "liked", type: "circle", source: "liked",
                   paint: { "circle-radius": 5, "circle-color": "#f0476a", "circle-stroke-color": "#14171c", "circle-stroke-width": 2 } });
      m.addSource("next", { type: "geojson", data: fc([]) });
      m.addLayer({ id: "next", type: "circle", source: "next",
                   paint: { "circle-radius": 5, "circle-color": "#e8820e", "circle-opacity": 0.35,
                            "circle-stroke-color": "#e8820e", "circle-stroke-width": 1, "circle-stroke-opacity": 0.7 } });
      setReady(true);
    });
    m.on("dragend", () => setMoved(true));
    const el = document.createElement("div");
    el.className = "map-pin";
    el.innerHTML = '<svg viewBox="0 0 40 52" width="40" height="52"><path d="M20 50S4 32.5 4 19a16 16 0 1 1 32 0c0 13.5-16 31-16 31z" fill="#e8820e" stroke="#fff" stroke-width="3"/><circle cx="20" cy="19" r="6" fill="#fff"/></svg>';
    pin.current = new maplibregl.Marker({ element: el, anchor: "bottom" });
    map.current = m;
    return () => m.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!ready) return;
    map.current.getSource("area").setData(circle(area.lat, area.lon, area.radius));
    if (!current) map.current.flyTo({ center: [area.lon, area.lat], zoom: 11.5, duration: 1200 });
    setMoved(false);
  }, [ready, area.lat, area.lon, area.radius]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!ready) return;
    map.current.getSource("next").setData(fc(upcoming));
    map.current.getSource("liked").setData(fc(liked));
  }, [ready, upcoming, liked]);

  useEffect(() => {
    if (!ready || !current) { pin.current?.remove(); return; }
    pin.current.setLngLat([current.lon, current.lat]).addTo(map.current);
    map.current.flyTo({ center: [current.lon, current.lat], zoom: Math.max(map.current.getZoom(), 13.5),
                        padding: { right: padRight, top: 80, bottom: padRight ? 40 : 360, left: 40 + padLeft }, duration: 1100, essential: true });
    setMoved(false);
  }, [ready, current, padRight, padLeft]);

  return (
    <div className="map-wrap">
      <div ref={box} className="map" />
      {moved && (
        <button className="search-here" onClick={() => {
          const c = map.current.getCenter();
          onSearchArea(c.lat, c.lng);
          setMoved(false);
        }}>Search this area</button>
      )}
    </div>
  );
}
