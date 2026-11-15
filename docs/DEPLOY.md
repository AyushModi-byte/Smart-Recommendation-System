# Running and deploying the demo

The demo is `app/app.py` (Gradio). It needs a **bundle**: four files exported from a
trained run. The bundle is about 88 MB and is not in git.

## 1. Build the bundle (on the GPU server)

```bash
CUDA_VISIBLE_DEVICES=0 python training/06_export_bundle.py --data data_v3 --run v7 --brain brain --out bundle
```

This writes `bundle/{weights.npz, pois.npz, users.npz, meta.json}` and checks that the
numpy inference ranks exactly like the trained torch model.

Copy it to the laptop (CMD, from the repo root):

```
scp -r dgx:btp/Smart-Recommendation-System/bundle .
```

`users.npz` contains real Flickr user ids. The app does not use them; remove the
`user_ids` array before publishing the bundle anywhere public.

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

- Place names are built from the two most common photo tags and are often generic ("Paris · France").
- Some Flickr image links are dead; those cards show without a picture.
- The mood mode is a heuristic (popularity plus cluster membership) and has no offline evaluation.
- Free Spaces sleep after a period without visitors and take about a minute to wake.
