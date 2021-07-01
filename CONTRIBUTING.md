# Contributing

Hi there! We're thrilled that you'd like to contribute to this project. Your help is 
essential for keeping it great.

Please note that this project is released with a [Contributor Code of Conduct]
[code-of-conduct]. By participating in this project you agree to abide by its terms.

## Never push modifications directly to do master branch, even if you have permissions

Several continuous integration workflows are configured for the repository, they 
need to pass successfully before considering merging modifications into the master 
branch.

## Submitting a pull request

1. Go to BCDI's repository on GitHub. Click on the “Fork” button. This will create a 
   fork of BCDI on your GitHub repository.
1. Click on the “Code” button. Select “HTTPS” is selected, unless you’ve set up SSH. 
   Click on the clipboard icon to copy the address.
1. Open a terminal. Navigate to a directory where you want to put your repository.
1. Clone your fork of BCDI to your local computer: `https://github.
   com/your-name/bcdi.git`. This will create a BCDI subdirectory inside the 
   directory you were already in.
1. Synchronize your local clone with the original repository (you can use a 
   different name from upstream, though it is a common usage): `git remote add 
   upstream https://github.com/carnisj/bcdi.git`
1. Check the remotes with `git remote -v`. There should be origin (which corresponds 
   to your fork) and upstream (which corresponds to the original repository)
1. Fetch the most recent changes from remotes: `git fetch --all`
1. Create a feature branch based off of the main branch on the upstream remote, and 
   give it a descriptive name. This command will create the branch, and switch you 
   to it: `git checkout -b branch-name upstream/main`
1. To check the branches that you have and which branch you are on, type: `git branch`
1. Push and link the branch to your fork on GitHub: `git push --set-upstream origin 
   branch-name`
1. Install the dependencies 
1. Edit a file, for example filename.py, add tests, and make sure the tests still pass.
1. Commit the file and push it to GitHub:
   - `git add filename.py`
   - `git commit -m "Add exciting new feature"`
   - `git push`
1. [submit a pull request][pr]
1. Wait for your pull request to be reviewed and merged.

Here are a few things you can do that will increase the likelihood of your pull 
request being accepted:

- Follow the [style guide][style] which is using PEP 8 recommendations.
- Run [black] [blck] against your code, with the default line length of 88 characters.
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

[pr]: https://docs.github.com/en/github/collaborating-with-pull-requests/
[style]: https://www.python.org/dev/peps/pep-0008/
[blck]: https://pypi.org/project/black/
[gcm]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[code-of-conduct]: CODE_OF_CONDUCT.md