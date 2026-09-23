# GitHub delivery

Repository: https://github.com/IamGRootMSE/personalization-ranking-engine

Project branch: `project/personalization-ranking-engine`.

The repository was empty at initial inspection. The project is committed locally and the delivery process attempts a normal non-force push to that branch. See the final delivery message for the verified remote outcome and commit.

## GitHub Pages setup

1. In the repository, open **Settings → Pages → Build and deployment**.
2. Select **GitHub Actions** as the source.
3. Allow the `github-pages` environment to deploy from the project branch if an environment branch rule restricts it.
4. Run **Deploy portfolio to GitHub Pages** from the Actions tab, selecting `project/personalization-ranking-engine` (or push a new commit to the branch).
5. Wait for `actions/deploy-pages` to succeed; use its emitted URL and verify the rendered site.

`pages.yml` builds from a fixed allowlist. It never trains MovieLens or uploads private model assets. `ci.yml` separately runs Python/Node tests and a small real-data training/evaluation smoke check. Smoke data and checkpoints remain in the runner's private workspace and are not uploaded.

To retry a blocked local push after signing into GitHub with your usual Git credential manager:

```bash
git push -u origin project/personalization-ranking-engine
```

Do not force-push or overwrite a branch that has acquired unrelated work. If the remote changes, fetch and reconcile before pushing. Neither workflow needs a repository secret or API key; Pages uses GitHub's scoped workflow token and OIDC. A workflow file alone is not proof of a live deployment.
