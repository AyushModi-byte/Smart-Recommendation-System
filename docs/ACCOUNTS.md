# User accounts (optional)

The swipe app works without accounts: likes, saves and posts stay in the browser. With a Supabase project
connected, people can sign in by email link, and then:

- likes, saves and skips sync to their account, so the feed follows them across devices (anything liked before
  signing in is merged in on first sign-in);
- a first-time profile asks for a name, a home area (the feed starts there) and moods they enjoy;
- a **Profile** tab shows liked count, posts, a taste summary (top scenery types among likes), the home area and
  their moods as one-tap filters;
- posts are stored on the server and shown to everyone in a **Community** tab.

Accounts switch on only when both settings below are present at build time; otherwise the app is unchanged.

## Setup (about 10 minutes)

1. Create a free project at [supabase.com](https://supabase.com).
2. **SQL Editor → New query**: paste [`supabase/schema.sql`](../supabase/schema.sql) and run it. It creates
   `profiles`, `place_actions` and `posts` with row-level security: each user reads and writes only their own
   data; posts and display names are public.
3. **Authentication → URL Configuration**: Site URL = `https://recommendation-system-gray-seven.vercel.app`;
   add `http://localhost:5173` under Redirect URLs for local development.
4. **Project Settings → API**: copy the Project URL and the `anon` public key.
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
| `web/src/AccountViews.jsx` | Sign-in, first-time profile, Profile tab, Community tab |
| `web/src/App.jsx` | Session handling, merge on first sign-in, sync on like / skip / save / undo / post |
| `supabase/schema.sql` | Tables and row-level security policies |
