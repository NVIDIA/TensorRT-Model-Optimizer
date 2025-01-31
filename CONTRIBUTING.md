# TensorRT Model Optimizer OSS Contribution Rules

This readme provides guidelines for writing and contributing code to the this repository. Make sure
that all `dev` optional requirements are installed with ModelOpt.

## Code linting

- All code (Python and C++) is auto-checked to adhere to the coding standards upon commit (see below for more info).
- Check out [`.pre-commit-config.yaml`](.pre-commit-config.yaml) for detailed information about each tool.
- If you would like to integrate the linting tools into your IDE, check out the
  documentation for the respective IDE, e.g., docs for [auto-formatting](https://code.visualstudio.com/docs/python/editing#_formatting) and
  [linting](https://code.visualstudio.com/docs/python/linting) in VSCode.
- For VSCode, we also provide default workspace settings, see [here](./.vscode/settings.json) for detailed instructions.

## Pre-commit hooks

Please enable pre-commit hooks as follows to automatically check / fix code quality before issues committing:

```bash
pre-commit install
```

If you simply want to add some temporary commit that skips the checks, you can use the `-n` flag during commit:

```bash
git commit -m "temporary commit" -n
```

If you want to run the pre-commit hooks without committing, you can use the following command:

```bash
pre-commit run --all-files
```

## Submitting your code

- Create a fork of the repository.
- Rebase (not merge) your code to the most recent commit of the main branch. We want to ensure linear history,
  check [Merge vs Rebase](https://www.atlassian.com/git/tutorials/merging-vs-rebasing). Remember to test again after rebase.

```bash
git pull
git rebase origin/main
git push origin <branch> --force-with-lease
```

- When pushing the rebased (or any) branch, use `git push --force-with-lease` instead of `git push --force`.
- Submit a pull request and assign at least two reviewers.
- Since there is no CI/CD process in place yet, the PR will be accepted and the corresponding issue closed only after
  adequate testing has been completed, manually, by the developer and/or TensorRT engineer reviewing the code.

## Signing your work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original
  work, or you have rights to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the Developer Certificate of Origin (DCO):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
