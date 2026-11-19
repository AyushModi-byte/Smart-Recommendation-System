# User accounts (optional)

The swipe app works without accounts: likes, saves and posts stay in the browser. With a Supabase project
connected, the app opens on a login page (email + password, or an emailed sign-in link, with "forgot password"
and a "continue without an account" option). A left sidebar holds Discover, Liked places, Saved for later, My
posts, People like you and Recent posts, with the signed-in user's profile and menu at the bottom. Signed in:

- **People like you**: other users ranked by how close their taste is to yours. A user's taste is the mean
  embedding of the places they liked, centred on the average place (`POST /api/taste`), and matches are by cosine
  similarity. Each match shows their recent likes you have not seen; tap one to open it. Only users who allow
  it in their profile are listed (`tastes` table; switching it off deletes the row).

- likes, saves and skips sync to their account, so the feed follows them across devices (anything liked before
  signing in is merged in on first sign-in);
- a first-time profile asks for a name, a home area (the feed starts there) and moods they enjoy;
- a **Profile** tab shows liked count, posts, a taste summary (top scenery types among likes), the home area and
  their moods as one-tap filters;
- posts are stored on the server and shown to everyone in a **Community** tab.

Accounts switch on only when both settings below are present at build time; otherwise the app is unchanged.

## Setup (about 10 minutes)

1. Create a free project at [supabase.com](https://supabase.com).
2. **SQL Editor → New query**: paste [`supabase/schema.sql`](../supabase/schema.sql) and run it, then the same
   with [`supabase/002_people.sql`](../supabase/002_people.sql), then
   [`supabase/003_grants.sql`](../supabase/003_grants.sql) (table privileges; without it the app shows
   "permission denied for table profiles"). They create `profiles`, `place_actions`, `posts`
   and `tastes` with row-level security: each user reads and writes only their own data; posts, display names
   and shared tastes are public.
3. **Authentication → URL Configuration**: Site URL = `https://recommendation-system-gray-seven.vercel.app`;
   add `http://localhost:5173` under Redirect URLs for local development.
4. **Project Settings → API Keys**: copy the Project URL and the **publishable** key (`sb_publishable_…`, or the
   legacy `anon` key). **Never the secret key (`sb_secret_…`) or `service_role`**: everything in a `VITE_`
   variable is compiled into the public JavaScript, and those keys bypass row-level security.
5. In Vercel (**Project → Settings → Environment Variables**) add
   `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`, then redeploy. For local development put the same two
   lines in `web/.env.local` (git-ignored).

The anon key is meant to be public; row-level security is what protects the data. Never put the
`service_role` key in the web app.

Supabase's built-in email sender is rate-limited (a few emails per hour), which is fine for a demo. For more,
set up custom SMTP under **Authentication → Emails**.

## Code

| File | What it does |
|---|---|
| `web/src/account.js` | Supabase client and data calls (profile, actions, posts) |
| `web/src/AccountViews.jsx` | Login page, password reset, first-time profile, Profile, Community, People like you |
| `web/src/Sidebar.jsx` | Left navigation and the account block at the bottom |
| `server/main.py` `/api/taste` | Taste vector of a list of liked places, for matching people |
| `web/src/App.jsx` | Session handling, merge on first sign-in, sync on like / skip / save / undo / post |
| `supabase/schema.sql`, `002_people.sql` | Tables and row-level security policies |
