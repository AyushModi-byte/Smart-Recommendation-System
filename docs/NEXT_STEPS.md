# Next steps

Where the project stands: the v2 model and every experiment in the proposal are done
([RESULTS.md](RESULTS.md)), and the swipe app is live ([SWIPE_APP.md](SWIPE_APP.md)). What follows is what is
left for the B.Tech submission, then directions beyond it.

## Left for the submission

| # | Item | Why | Effort |
|---|---|---|---|
| 1 | **Live weather and crowd in the swipe app.** Fetch current weather for the area (Open-Meteo, no key); on rain or cold, show indoor-leaning places first and say so on the card ("Raining now: indoor places first"). Add a "quieter places" switch from the crowd estimate already in the bundle. | The proposal's Phase 3 promises real-time weather and traffic filtering; only the Gradio demo has it. Offline it does not change scores, so it is a user-facing filter, as mood already is. | Half a day |
| 2 | **User accounts and profiles.** Sign-in, onboarding (location + a few moods), feed centred on the user's location, profile page (liked places on a map, posts, taste summary), likes and posts stored server-side so they sync and others can see posts. | Each user gets their own feed, and the app becomes a real social product. Also gives real like/skip data to measure the recommender live. | 2–3 days |
| 3 | **Report and slides.** Replace the v1 numbers (AUC 0.998 and the rest are invalid, see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)) with the v2 results, the "what each idea contributed" table, the proposal checklist with the two honest misses (MAE 52% vs 60%, new city only with a destination), and the live app with screenshots. | `BTP docs/` still holds the April versions. | 1–2 days |
| 4 | **Social influence clusters** (optional). Cluster users by the context of their visits (time, weather, crowd) and report the groups. | The one proposal item not built. Expect insight for the report, not a ranking gain. | Half a day |

Housekeeping: revoke the GitHub token that was shared in chat, change the DGX password, back up `brain/`, and
clear finished runs on the DGX (about 36 of 50 GB used).

## Future directions

**Recommendation quality**
- **Learn online from likes and skips.** Feedback events are already logged per session. With accounts, retrain
  or fine-tune the taste blend on real swipes, and compare variants with A/B tests on like rate.
- **Sequence-aware model.** A transformer over the visit history (SASRec-style) instead of mean pooling, to
  capture order ("museum, then dinner nearby").
- **Better visual encoder.** CLIP-style image embeddings instead of ResNet50 would also allow text search
  ("quiet lakeside at sunset") against the same place vectors.
- **New-city trips without a destination.** Currently no better than popularity. Predicting *where* someone
  goes next (city level) before *which place* is a separate model worth trying.
- **Fresher data.** The photos are 2004–2014. Adding recent Flickr or Wikimedia Commons photos, and checking
  places against OpenStreetMap, would drop closed places and add new ones.

**Product**
- **Trip planning.** Turn liked places into a day route (ordering, walking time, opening hours).
- **Group mode.** Two or more people swipe; recommend places that fit everyone.
- **Social graph.** Follow people with similar taste, "liked by people like you", posts with photos.
- **Mobile app.** The web app already works at phone width; a PWA (installable, offline cache) is the cheap
  next step, React Native the full one.

**Engineering**
- Move the bundle to a proper database (Postgres + pgvector) so places can be updated without a redeploy.
- Monitoring: request latency, error rate and like rate per release.
- Paid always-on hosting, or a scheduled ping, to avoid the free tier's cold start before demos.
