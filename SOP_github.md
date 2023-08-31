# Standard Operating Procedure for using GitHub from ZERO
## PLEASE FOLLOW THIS FIRST BEFORE SOP.md

## Install Git
### For MacOS
#### Installing homebrew
1. Open Terminal.
2. Paste this into the terminal: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
#### Installing Git
3. Paste this into the terminal: `brew install git`

### For Windows
1. Go to https://git-scm.com/download/win.
2. Download the 64-bit Git for Windows Setup.
3. Follow the instructions and install Git.

## Sign up on GitHub
1. Go to https://github.com/ and make an account!

## Setup personal access token
1. Login to GitHub.
2. Go to Settings > Developer Settings (at bottom) > Personal access tokens > Tokens (classic).  Name the token, check all the boxes, and press the generate token button. Copy the generated token to somewhere where you won't lose it.

## Download the GitHub repository
1. In the terminal (or called command prompt), go to the directory you want to store your repository. For example, `C:\Users\Azalea\University of Toronto\Helen Tran - 2020 Azalea Uva\Coding`
2. Type `cd spectrometry-visualizer` in the terminal.
3. Type `git clone git@github.com:teamtran/spectrometry-visualizer.git` in the terminal.
4. When prompted to input your github information.
    - Username: type in your github username
    - Password: copy and paste your personal access token that was generated in the previous instructions.
5. If authorized and done correctly, the files should be downloaded to your directory.

You can now move on to the SOP.md file