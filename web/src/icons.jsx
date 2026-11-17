const S = ({ children, size = 20, fill = "none", ...p }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor" strokeWidth="2"
       strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...p}>{children}</svg>
);

export const Pin = (p) => <S {...p}><path d="M12 22s7-6.2 7-12a7 7 0 1 0-14 0c0 5.8 7 12 7 12z" /><circle cx="12" cy="10" r="2.5" /></S>;
export const Heart = (p) => <S fill="currentColor" stroke="none" {...p}><path d="M12 21s-7.5-4.6-9.6-9.3C.9 8.2 3.2 4.5 6.9 4.5c2.1 0 3.6 1.1 5.1 3 1.5-1.9 3-3 5.1-3 3.7 0 6 3.7 4.5 7.2C19.5 16.4 12 21 12 21z" /></S>;
export const X = (p) => <S {...p}><path d="M6 6l12 12M18 6L6 18" /></S>;
export const Bookmark = (p) => <S {...p}><path d="M6 3h12v18l-6-4-6 4z" /></S>;
export const Undo = (p) => <S {...p}><path d="M9 14L4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" /></S>;
export const Expand = (p) => <S {...p}><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" /></S>;
export const Spark = (p) => <S {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6" /></S>;
export const Search = (p) => <S {...p}><circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" /></S>;
export const Camera = (p) => <S {...p}><path d="M4 7h3l2-3h6l2 3h3v13H4z" /><circle cx="12" cy="13" r="3.5" /></S>;
export const Calendar = (p) => <S {...p}><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M3 10h18M8 3v4M16 3v4" /></S>;
export const Close = X;
export const Target = (p) => <S {...p}><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="2" /><path d="M12 1v3M12 20v3M1 12h3M20 12h3" /></S>;
