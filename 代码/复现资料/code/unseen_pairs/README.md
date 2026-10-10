# Supplementary unseen-pair experiments

From the package root, use the same dependencies and MPC_DATA_ROOT layout as the main README:

```
python code/unseen_pairs/run_experiment.py
python code/unseen_pairs/validate_protocol.py
python code/unseen_pairs/analyze_results.py
```

Results are saved in supplement/unseen_pairs. Running the experiment overwrites those results. Archived protocol/completion hashes refer to the original experiment workspace before manuscript integration; the solver hash is unchanged. The paper has subsequently been edited to incorporate the results. No raw datasets are distributed. The main README explains dataset paths and providers.

For a separate rerun directory, set MPC_UNSEEN_OUT before starting Python. Both run_experiment.py and analyze_results.py use it. validate_protocol.py additionally needs protocol.json in that directory. The experiment can run in a standalone code repository without manuscript files; if main.tex/main.pdf are present, their hashes are also protected during the run. Use python code/verify_unseen_results.py to verify the delivered archived results without raw data.
