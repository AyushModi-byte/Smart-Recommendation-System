-- AuraPath: "people with similar taste". Run once in the Supabase SQL editor, after schema.sql. Safe to re-run.
--
-- tastes holds what other users may see: a taste vector (the centred mean embedding of the places a user liked,
-- computed by the API), their latest liked places and top scenery types. A row exists only while the user's
-- profile is discoverable; turning it off deletes the row. Liked places themselves stay private (place_actions).

alter table public.profiles add column if not exists discoverable boolean not null default true;

create table if not exists public.tastes (
  user_id         uuid primary key references public.profiles on delete cascade default auth.uid(),
  taste           real[] not null,
  favorites       jsonb not null default '[]',      -- up to 8 recently liked places: {id, name, town, country, category, photo}
  liked_count     integer not null default 0,
  top_categories  text[] not null default '{}',
  updated_at      timestamptz not null default now()
);

alter table public.tastes enable row level security;

drop policy if exists "tastes are public" on public.tastes;
drop policy if exists "own taste insert"  on public.tastes;
drop policy if exists "own taste update"  on public.tastes;
drop policy if exists "own taste delete"  on public.tastes;
create policy "tastes are public" on public.tastes for select using (true);
create policy "own taste insert"  on public.tastes for insert with check (auth.uid() = user_id);
create policy "own taste update"  on public.tastes for update using (auth.uid() = user_id);
create policy "own taste delete"  on public.tastes for delete using (auth.uid() = user_id);
