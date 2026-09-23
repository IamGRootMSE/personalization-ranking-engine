# GitHub delivery

Repository: https://github.com/IamGRootMSE/personalization-ranking-engine

Project branch: `project/personalization-ranking-engine`.

The repository was empty at initial inspection. The project is committed locally. Publication was attempted on 2026-09-23 and is **blocked**; no remote project branch or live site was verified.

Two available routes were checked:

- Command-line Git had no usable GitHub credentials. A noninteractive push reported `fatal: could not read Username for 'https://github.com': terminal prompts disabled` after the configured credential manager failed to provide authentication.
- The connected GitHub integration could read repository metadata, but its repository initialization write returned HTTP 403: `Resource not accessible by integration`. The failed request did not create a file or remote commit. Repository metadata indicating account-level push permission did not establish the integration's actual write scope.

To finish, sign in with a Git credential manager that has permission to push repository contents **and workflow files**, then run the push command below from this local checkout. Alternatively reconnect the GitHub integration with access to this repository and the required contents/workflow permissions. Credentials must not be pasted into source files or committed. GitHub CI could not execute because the branch was not published; the corresponding local tests, smoke training, and site build passed.

The delivered environment's bundled Git required a local `http.sslBackend=openssl` setting and `GIT_EXEC_PATH` pointing to its `mingw64/bin` folder to locate HTTPS helpers. These environment repairs are not tracked repository configuration, do not disable TLS verification, and are normally unnecessary with a standard Git installation.

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
