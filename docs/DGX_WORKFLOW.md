# Working between the laptop and the DGX

**Rule of thumb:** code moves through git, data moves through `scp`, results come back as small JSON.

| | Laptop (Windows) | DGX |
|---|---|---|
| Role | edit code, review results, run the demo | data prep, training, evaluation |
| Holds | the only full copy of `brain/` | the ~12 GB subset it needs, `data_v*/`, `runs/` |
| Never | train on full data | be the only place a result lives |

## One-time setup

1. **SSH alias.** Add to `C:\Users\<you>\.ssh\config`:

   ```
   Host dgx
       HostName <server address>
       User <your username>
       ServerAliveInterval 60
   ```

2. **Key login** (so commands run without a password prompt):

   ```
   ssh-keygen -t ed25519
   type %USERPROFILE%\.ssh\id_ed25519.pub | ssh dgx "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
   ssh dgx hostname
   ```

3. **Code and environment on the server:**

   ```
   ssh dgx "git clone https://github.com/AyushModi-byte/Smart-Recommendation-System.git"
   ssh dgx "bash Smart-Recommendation-System/scripts/dgx_setup.sh"
   ```

4. **Data** (about 12 GB, run from the repo root on the laptop):

   ```
   scripts\dgx_push_data.cmd
   ```

## Daily loop

1. Change code on the laptop, commit, `git push`.
2. On the server: `git pull`, then start the job with `nohup` so it survives a dropped
   connection (this server has no `tmux`):

   ```
   ssh dgx
   cd btp/Smart-Recommendation-System && source ~/aura-env/bin/activate
   nvidia-smi                       # pick a free GPU
   CUDA_VISIBLE_DEVICES=<n> nohup python training/01_train.py --data data_v3 --name <run> > logs/<run>.log 2>&1 &
   tail -f logs/<run>.log
   ```

3. Bring results back to the laptop:

   ```
   scripts\dgx_pull_results.cmd
   ```

   This copies `runs/*/*.json` (config, history, test results). Weights stay on the
   server unless a run is chosen for the demo.
4. Commit the JSON for runs worth keeping. `runs/**/*.pth` and logs are git-ignored.

## Experiment discipline

Start every new experiment with `--smoke` (1–2 minutes) before the full run, give
each run a unique `--name`, and never tune on the test set (`02_evaluate.py` is
run once per chosen model).

## Shared-server manners

- Check `nvidia-smi` and use one GPU via `CUDA_VISIBLE_DEVICES`.
- Keep the virtual environment inside the project folder.
- Do not store the GitHub token in the remote URL; use `gh auth login` or an SSH key.
