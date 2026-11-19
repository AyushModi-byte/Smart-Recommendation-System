-- Table privileges for the API roles. Newer Supabase projects no longer grant these automatically to
-- tables created in SQL, which shows up in the app as "permission denied for table profiles".
-- Row-level security (schema.sql, 002_people.sql) still decides which rows each user can touch.
-- Safe to run more than once.

grant usage on schema public to anon, authenticated;

grant select                         on public.profiles      to anon, authenticated;
grant insert, update                 on public.profiles      to authenticated;

grant select, insert, update, delete on public.place_actions to authenticated;

grant select                         on public.posts         to anon, authenticated;
grant insert, delete                 on public.posts         to authenticated;

grant select                         on public.tastes        to anon, authenticated;
grant insert, update, delete         on public.tastes        to authenticated;

grant usage, select on all sequences in schema public to authenticated;
