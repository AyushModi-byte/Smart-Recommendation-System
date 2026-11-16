# Running and deploying the demo

The demo is `app/app.py` (Gradio). It needs a **bundle**: four files exported from a
trained run. The bundle is about 88 MB and is not in git.

## 1. Build the bundle (on the GPU server)

```bash
CUDA_VISIBLE_DEVICES=0 python training/06_export_bundle.py --data data_v4 --run p1_split --brain brain --out bundle --places _v2
```

This writes `bundle/{weights.npz, pois.npz, users.npz, meta.json}` and checks that the
numpy inference ranks exactly like the trained torch model.

Copy it to the laptop (CMD, from the repo root):

```
scp -r dgx:btp/Smart-Recommendation-System/bundle .
```

The bundle contains no user ids; sample travellers are stored by index only.

To refresh only the place names of an existing bundle (no GPU needed):

```bash
python training/poi_names.py --brain brain --bundle bundle
```

## 2. Run locally

```
uv venv .venv --python 3.10
uv pip install --python .venv\Scripts\python.exe numpy plotly gradio
.venv\Scripts\python.exe app\app.py
```

Open the address it prints (http://127.0.0.1:7860). Start-up takes about 10 seconds
and uses under 1 GB of memory.

## 3. Deploy free on Hugging Face Spaces

1. Create an account at huggingface.co, then **New Space**: SDK *Gradio*, hardware *CPU basic (free)*.
2. Create a **write** access token (Settings, Access Tokens).
3. From the repo root in CMD:

   ```
   uv pip install --python .venv\Scripts\python.exe huggingface_hub
   .venv\Scripts\hf.exe auth login
   .venv\Scripts\hf.exe upload <username>/<space-name> app . --repo-type space
   .venv\Scripts\hf.exe upload <username>/<space-name> bundle bundle --repo-type space
   ```

   The first upload sends `app.py`, `aurapath_np.py`, `requirements.txt` and the Space
   `README.md` (its header configures the Space). The second sends the bundle into a
   `bundle/` folder next to `app.py`, where the app looks for it.
4. The Space builds and starts on its own. The public address is
   `https://huggingface.co/spaces/<username>/<space-name>`.

To update the app later, repeat the first `upload` command. To ship a new model,
rebuild the bundle and repeat the second.

## Known limits of the demo

- Place names are built from photo tags (`training/poi_names.py`). Popular places read well ("Eiffel Tower, Paris"); about a fifth of the rarely visited ones only get "Near <town>".
- Some Flickr image links are dead; those cards show without a picture.
- Mood, scenery, live conditions and avoid-crowds are explicit filters, not learned ranking signals: offline, time, weather and crowd level did not improve accuracy, and mood has no ground truth.
- Live weather comes from Open-Meteo; if the call fails the app falls back to the manual controls.
- The export packs mood and context profiles only if `10_mood_profile.py` and `weather_noaa.py` have been run.
- Free Spaces sleep after a period without visitors and take about a minute to wake.
