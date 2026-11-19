-- AuraPath accounts: run once in the Supabase SQL editor (Dashboard > SQL Editor > New query).
-- Row-level security: a user reads and writes only their own profile data and actions; posts and
-- profile names are readable by everyone, so the community feed can show who wrote what.

create table if not exists public.profiles (
  id          uuid primary key references auth.users on delete cascade,
  name        text not null check (char_length(name) between 1 and 40),
  home        jsonb,                                  -- {name, lat, lon, radius}: where the feed starts
  moods       text[] not null default '{}',
  created_at  timestamptz not null default now()
);

create table if not exists public.place_actions (
  user_id     uuid not null references public.profiles on delete cascade default auth.uid(),
  place_id    integer not null,
  action      text not null check (action in ('accept', 'reject', 'save')),
  card        jsonb,                                  -- the card as shown, for the profile page
  created_at  timestamptz not null default now(),
  primary key (user_id, place_id, action)
);

create table if not exists public.posts (
  id          bigint generated always as identity primary key,
  user_id     uuid not null references public.profiles on delete cascade default auth.uid(),
  place_id    integer not null,
  place       jsonb not null,
  body        text not null check (char_length(body) between 1 and 280),
  created_at  timestamptz not null default now()
);
create index if not exists posts_created_idx on public.posts (created_at desc);

alter table public.profiles      enable row level security;
alter table public.place_actions enable row level security;
alter table public.posts         enable row level security;

create policy "profiles are public"       on public.profiles for select using (true);
create policy "own profile insert"        on public.profiles for insert with check (auth.uid() = id);
create policy "own profile update"        on public.profiles for update using (auth.uid() = id);

create policy "own actions read"          on public.place_actions for select using (auth.uid() = user_id);
create policy "own actions insert"        on public.place_actions for insert with check (auth.uid() = user_id);
create policy "own actions update"        on public.place_actions for update using (auth.uid() = user_id);
create policy "own actions delete"        on public.place_actions for delete using (auth.uid() = user_id);

create policy "posts are public"          on public.posts for select using (true);
create policy "own posts insert"          on public.posts for insert with check (auth.uid() = user_id);
create policy "own posts delete"          on public.posts for delete using (auth.uid() = user_id);
