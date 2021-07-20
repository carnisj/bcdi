# Contributing

Hi there! We're thrilled that you'd like to contribute to this project. Your help is 
essential for keeping it great.

Please note that this project is released with a 
[Contributor Code of Conduct][code-of-conduct]. By participating in this project 
you agree to abide by its terms.

### *Never push changes directly to the master branch, even if you have permissions.*

Several continuous integration workflows are configured for the repository, they 
need to pass successfully before considering merging modifications into the master 
branch.

## 1. Create your fork and local clone of the repository

- Go to BCDI's repository on GitHub. Click on the “Fork” button. This will create a 
   fork of BCDI on your GitHub repository.
- Click on the “Code” button. Select “HTTPS” is selected, unless you’ve set up SSH. 
   Click on the clipboard icon to copy the address.
- Open a terminal. Navigate to a directory where you want to put your repository.
- Clone your fork of BCDI to your local computer: `gti clone https://github.com/your-name/bcdi.git`.
  This will create a BCDI subdirectory inside the directory you were already in.
  
## 2. Make your local clone track the original upstream repository.

(This is optional but useful if you plan to make more than one contribution)

- Synchronize your local clone with the original repository (you can use a 
   different name from upstream, though it is a common usage): `git remote add 
   upstream https://github.com/carnisj/bcdi.git`
- Check the remotes with `git remote -v`. There should be origin (which corresponds 
   to your fork) and upstream (which corresponds to the original repository)
  
## 3. Update your local clone with the latest changes from the upstream repository

- Be sure that you are on your master branch: `git checkout master`
- Fetch the most recent changes from remotes: `git fetch --all`
- Merge the upstream changes into your own project: `git merge upstream/master` 
- Push the changes to your fork on GitHub: `git push origin master`

## 4. Work on your new feature and create a pull request

***"Short-lived, single feature branches are the way to go."***

- Do not work on your master branch, but create a feature branch and give it a 
  descriptive name. This command will create the branch, and switch you to it:
  `git checkout -b branch-name`
- To check the branches that you have and which branch you are on, type: `git branch`
- Push and link the branch to your fork on GitHub: `git push --set-upstream origin 
   branch-name`
- Edit a file, for example filename.py, add tests, and make sure the tests still pass.
- Run automated tasks: in a terminal, go to the same level as dodo.py and simply 
  type: `doit`. This will run tests with coverage and black (code formatting).
- Commit the file and push it to GitHub:
   - `git add filename.py`
   - `git commit -m "Add exciting new feature"`
   - `git push`
   
## 5. Create a pull request

- If they were several changes committed to the upstream master branch while you 
  were working on your feature branch, you may want/need to [rebase your branch][rb] 
  (to get a clean history). For this, you need to:
  - Update your local master branch (section 3). This should be a simple 
    fast-forward merge since you never committed to your master branch. If you did 
    commit to your master branch, see * below.
  - Rebase your feature branch on your master branch (note that there may be 
    conflicts, if the upstream master changes affect the same portion of code as 
    yours):
   - `git checkout branchname`
   - `git rebase master`
   - `git push`
- [Submit a pull request][pr].
- Wait for your pull request to be reviewed and merged.
  
\* If you committed to your master branch, you may consider first rebasing it on the 
upstream master: 
 - `git fetch upstream`
 - `git checkout master`
 - `git rebase upstream/master`
 - `git push -f origin master`

## 6. Delete your feature branch and start over

- Once your pull request has been merged, delete the local and remote branches:
  - Switch to your local master branch: `git checkout master`
  - Delete the remote branch: `git push origin :branch-name`
  - Delete the local branch: `git branch -D branch-name`
  - You may have to clean the stale branch: `git remote prune origin`
- Start over at section 3.

## Here are a few things you can do that will increase the likelihood of your pull request being accepted:

- Follow the [style guide][style] (PEP 8 recommendations).
- Write and update tests.
- Keep your change as focused as possible. If there are multiple changes you would 
  like to make that are not dependent upon each other, consider submitting them as 
  separate pull requests.
- Write a [good commit message][gcm].

Work in Progress pull requests are also welcome to get feedback early on, or if 
there is something that blocked you.

## Resources

- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [Using Pull Requests](https://help.github.com/articles/about-pull-requests/)
- [GitHub Help](https://help.github.com)

[rb]: https://git-scm.com/book/en/v2/Git-Branching-Rebasing
[pr]: https://docs.github.com/en/github/collaborating-with-pull-requests/
[style]: https://www.python.org/dev/peps/pep-0008/
[blck]: https://pypi.org/project/black/
[gcm]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[code-of-conduct]: CODE_OF_CONDUCT.md
