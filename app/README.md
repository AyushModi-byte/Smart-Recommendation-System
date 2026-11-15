---
title: AuraPath
emoji: 🗺️
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: Place recommendations learned from geotagged photos
---

# AuraPath demo

Pick a destination and a mood, or a sample traveller, and get places ranked by a model
trained on 3.7 million geotagged Flickr photos (YFCC100M, USA and Europe).

- **Pick a mood**: popularity blended with one of six aesthetic clusters.
- **Sample traveller, at this destination**: popularity blended with the trained model's score for that traveller.
- **Sample traveller, near places they know**: the model alone, over all places.

Inference is numpy-only. Source and evaluation: <https://github.com/AyushModi-byte/Smart-Recommendation-System>

Photos are Creative Commons images hosted by Flickr; each links to its source page.
