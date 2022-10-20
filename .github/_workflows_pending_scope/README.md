# Workflows pending the `workflow` OAuth scope

To activate CI on this repo:

```bash
gh auth refresh -h github.com -s workflow
git mv .github/_workflows_pending_scope/ci.yml .github/workflows/ci.yml
git commit -m "ci: activate CI pipeline" .github/workflows/
git push
```
