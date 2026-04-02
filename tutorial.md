# Step-by-Step Remote Server Tutorial

**Audience:** Students who are new to working on a remote Linux server  
**Format:** 1-2 sessions x 60 minutes, or self-paced at home  
**Authentication:** **TBD** (replace all placeholders once course staff confirms the login method)

---

## 0. Before you start

Have these ready on your **local machine**:

- your Andrew ID or school username
- server hostname: `TBD_SERVER_HOST`
- authentication method: `TBD`
- VS Code installed
- Git installed
- a terminal app
- Python 3 installed locally (optional, but useful if you want to generate SSH keys)
- **Note**: If you go through this tutorial at home, please install CMU VPN using this tutorial: https://www.cmu.edu/computing/services/endpoint/network-access/vpn/how-to/index.html
### Placeholders used in this tutorial

Replace these with real values when course staff provides them:

- `YOUR_ANDREW_ID`
- `TBD_SERVER_HOST`
- `YOUR_GITHUB_USERNAME`

---

## 1. Connect to the server

This section shows both the **big picture** and the **step-by-step commands**.  
Even if your course ends up using a different authentication method, it is still useful to understand how SSH keys and SSH config work.

### 1.1 What is SSH?

SSH (Secure Shell) is the standard way to connect securely to a remote server from your computer.

When you SSH into a server:

- you open a terminal session on the remote machine
- commands run on the server, not on your laptop
- files you edit are stored on the server

Basic SSH command:

```bash
ssh YOUR_ANDREW_ID@TBD_SERVER_HOST
```

---

### 1.2 SSH keys: what they are and why they matter

SSH key authentication uses a **key pair**:

- **private key**: stays on your computer and should never be shared
- **public key**: can be placed on the server so the server can verify your identity

Why use SSH keys?

- safer than retyping passwords all the time
- more convenient for repeated logins
- works well with VS Code Remote SSH, Git, and port forwarding

### 1.3 Generate an SSH key pair on your local machine

Run this on **your laptop**, not on the remote server:

```bash
ssh-keygen -t ed25519 -C "YOUR_ANDREW_ID@school.edu"
```

What happens next:

1. it asks where to save the key  
   - press **Enter** to use the default path: `~/.ssh/id_ed25519`
2. it asks for a passphrase  
   - recommended: use one
3. it creates two files:
   - `~/.ssh/id_ed25519` → **private key**
   - `~/.ssh/id_ed25519.pub` → **public key**

### 1.4 View your public key

```bash
cat ~/.ssh/id_ed25519.pub
```

You can share the **public** key with the server admin if needed.

Do **not** share this file:

```bash
~/.ssh/id_ed25519
```

That is your private key.

### 1.5 Add your public key to the server

If the server supports SSH key login, one common approach is:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub YOUR_ANDREW_ID@TBD_SERVER_HOST
```

If `ssh-copy-id` is not available, you may need to send your public key to course staff, or use a manual command like this:

```bash
cat ~/.ssh/id_ed25519.pub | ssh YOUR_ANDREW_ID@TBD_SERVER_HOST 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
```

> **Note:** The actual authentication workflow for this class is still **TBD**.  
> You may end up using school credentials, Duo, a password, SSH keys, or some combination of them.

### 1.6 Recommended SSH file permissions

Sometimes SSH refuses to use your key if permissions are too open.

On your local machine, these are safe defaults:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

If you use an SSH config file, you can also do:

```bash
chmod 600 ~/.ssh/config
```

### 1.7 Create an SSH config alias

An SSH config file lets you use a short nickname instead of typing the full host every time.

Open or create your config file:

```bash
nano ~/.ssh/config
```

Add this block:

```sshconfig
Host myclassserver
    HostName TBD_SERVER_HOST
    User YOUR_ANDREW_ID
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

What these lines mean:

- `Host myclassserver` → your shortcut name
- `HostName` → actual server address
- `User` → your login name
- `IdentityFile` → which SSH key to use
- `ServerAliveInterval` and `ServerAliveCountMax` → help reduce disconnects during idle periods

Now you can connect with:

```bash
ssh myclassserver
```

instead of:

```bash
ssh YOUR_ANDREW_ID@TBD_SERVER_HOST
```

### 1.8 Connect from the terminal

If no SSH config is set up yet:

```bash
ssh YOUR_ANDREW_ID@TBD_SERVER_HOST
```

If config is set up:

```bash
ssh myclassserver
```

If connection succeeds, your prompt will change and you will now be on the server.

A typical prompt may look something like:

```bash
YOUR_ANDREW_ID@servername:~$
```

### 1.9 Quick troubleshooting for SSH

See which key files you have:

```bash
ls -lah ~/.ssh
```

Run SSH in verbose mode if something fails:

```bash
ssh -v YOUR_ANDREW_ID@TBD_SERVER_HOST
```

Common issues:

- wrong hostname
- wrong username
- public key not added to server
- wrong permissions on `~/.ssh` or key files
- trying to use the private key from the wrong location
- school VPN required before login

---

### 1.10 Connect with VS Code

VS Code is useful if you want a graphical editor while still working on the server.

#### One-time setup

1. Install **Remote - SSH** from extensions in VSCode
2. Open the Command Palette
3. Choose **Remote-SSH: Open SSH Configuration File**
4. Add the same host block from `~/.ssh/config`

#### Connect

1. Open Command Palette
2. Choose **Remote-SSH: Connect to Host**
3. Select `myclassserver` or enter `YOUR_ANDREW_ID@TBD_SERVER_HOST`
4. Wait for a new VS Code window to open
5. Open a folder on the server

Inside that VS Code window:

- the file explorer shows files on the server
- the terminal runs on the server
- saving a file saves it directly to the server

---

### 1.11 Optional: Jupyter tunnel

If you want to run Jupyter on the server but use it in your local browser, forward a port.

First, on the **server**:

```bash
jupyter notebook --no-browser --port 8888
```

Then, from a **new local terminal** on your laptop:

```bash
ssh -L 8888:localhost:8888 YOUR_ANDREW_ID@TBD_SERVER_HOST
```

If you use your SSH config alias:

```bash
ssh -L 8888:localhost:8888 myclassserver
```

Then open this in your browser:

```text
http://localhost:8888
```

> If you use VS Code Remote SSH with Jupyter support, VS Code often handles port forwarding for you.

---

## 2. Create your workspace

Create a folder inside your own home directory on the server.

> `~` means “your home directory.”  
> You want a subfolder named after your Andrew ID:

```bash
mkdir -p ~/YOUR_ANDREW_ID/tutorial-demo
cd ~/YOUR_ANDREW_ID/tutorial-demo
pwd
ls -lah
```

Useful navigation commands:

```bash
cd ~
cd ..
cd -
pwd
ls -lah
```

---

## 3. Clone the GitHub repo

If the repo is public and students only need to **download and run** it, they can clone directly.

```bash
git clone https://github.com/tienvu95/remote_server_tutorial_demo.git
cd remote_server_tutorial_demo
```

### 3.1 When should students fork?

If students need to **make edits and push their own changes**, the cleanest beginner workflow is:

1. fork the repo on GitHub, the repo link https://github.com/tienvu95/remote_server_tutorial_demo.git
2. clone **their fork**
3. create a new branch
4. edit files
5. commit
6. push

That way, they do not need write access to the instructor repo.

### 3.2 Create a branch

```bash
git checkout -b my-first-edit
```

### 3.3 Create and inspect a text file

```bash
echo "Hello, this is MY_NAME." > student_notes.txt
cat student_notes.txt
nano student_notes.txt
```

In `nano`:

- type normally to edit
- press `Ctrl+O` then `Enter` to save
- press `Ctrl+X` to exit

You can inspect the file again with:

```bash
cat student_notes.txt
head student_notes.txt
tail student_notes.txt
```

### 3.4 Commit and push

```bash
git add student_notes.txt
git status
git commit -m "Add student notes"
git push origin my-first-edit
```

---

## 4. Set up a virtual environment

Create a project-specific Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
pip install --upgrade pip
pip install -r requirements.txt
```

When the environment is active, your prompt usually shows something like:

```bash
(.venv)
```

To leave the environment:

```bash
deactivate
```

### 4.1 Why not use one shared environment?

Because on a shared server:

- package versions may conflict
- one student's install can break another student's work
- some projects need different dependencies
- isolated environments make your work easier to reproduce later

---

## 5. Run code

### 5.1 Command line

Activate the environment and run the simple script:

```bash
source .venv/bin/activate
python hello_remote.py
```

### 5.2 Jupyter notebook

If your repo includes a notebook:

```bash
jupyter notebook --no-browser --port 8888
```

Open the notebook file:

```text
remote_demo.ipynb
```

If needed, pair this with the SSH tunnel from Section 1.11.

---

## 6. Transfer files

You will demo both `scp` and `rsync`.

> These commands are usually run from your **local machine**, not from inside the remote SSH session.

### 6.1 Download from server to local with `scp`

Run on your local machine:

```bash
scp YOUR_ANDREW_ID@TBD_SERVER_HOST:~/tutorial-demo/remote_server_tutorial_demo/student_notes.txt .
```

If you use SSH config:

```bash
scp myclassserver:~/tutorial-demo/remote_server_tutorial_demo/student_notes.txt .
```

### 6.2 Upload from local to server with `scp`

```bash
scp sample_local_upload.txt YOUR_ANDREW_ID@TBD_SERVER_HOST:~/tutorial-demo/remote_server_tutorial_demo/
```

or with config alias:

```bash
scp sample_local_upload.txt myclassserver:~/tutorial-demo/remote_server_tutorial_demo/
```

### 6.3 Use `rsync`

Download:

```bash
rsync -avz --progress YOUR_ANDREW_ID@TBD_SERVER_HOST:~/tutorial-demo/remote_server_tutorial_demo/student_notes.txt .
```

Upload:

```bash
rsync -avz --progress sample_local_upload.txt YOUR_ANDREW_ID@TBD_SERVER_HOST:~/tutorial-demo/remote_server_tutorial_demo/
```

With SSH config alias:

```bash
rsync -avz --progress sample_local_upload.txt myclassserver:~/tutorial-demo/remote_server_tutorial_demo/
```

### 6.4 Why show both `scp` and `rsync`?

- `scp` is simple and good for one file or a quick copy
- `rsync` is better for repeated syncs and larger folders because it only transfers changes

---

## 7. Helpful commands

### 7.1 Check files and directories

```bash
pwd
ls -lah
cd
mkdir
rm
cp
mv
cat
head
tail
```

### 7.2 Monitor resource usage

```bash
htop
```

If `htop` is unavailable:

```bash
top
```

### 7.3 Run a long job in `tmux`

Start a session:

```bash
tmux new -s demo
```

Run the script:

```bash
python long_job.py
```

Detach without stopping the job:

```text
Ctrl-b then d
```

List sessions:

```bash
tmux ls
```

Reconnect:

```bash
tmux attach -t demo
```

Kill the session when done:

```bash
tmux kill-session -t demo
```

### 7.4 Save logs

```bash
python long_job.py > demo.log 2>&1
cat demo.log
tail -f demo.log
```

### 7.5 Editors

- `nano file.txt` for beginners
- `vim file.txt` if already comfortable

---

## 8. Stuff to talk about

### 8.1 Slurm / scheduler

### 8.2 CPU vs GPU

Use CPU for:

- normal shell work
- data cleaning
- simple scripts
- lightweight analysis

Use GPU when:

- your code is written to use GPU libraries
- you are training larger ML or deep learning models
- the workload actually benefits from parallel hardware acceleration

### 8.3 Managing shared resources

Be a good citizen on a shared server:

- do not hog compute
- do not leave unnecessary jobs running
- be careful with private or restricted data
- do not edit, move, or delete other people's files
- stay organized in your own directories
- check memory / CPU usage before launching heavy jobs

---

