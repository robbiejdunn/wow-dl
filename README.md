# wow-dl

# Running

Build docker image e.g.

`docker build -t wowrlagents .`

Run with the following command:

`docker run`

Go to localhost:6080. Open a terminal and create a directory to hold game screenshots:

`mkdir data`

Run WoW in wine, when prompted to install mono and gecko- accept:

`wine client/WoW.exe`

Login as Dizia on robbie's account. Make sure the chatbox in game isn't open, run the training:

```
source venv/bin/activate
python -m rl_agents
```

NOTE: it seems you can close the novnc window but opening a new one appeared to crash it. Best to observe in game from another account or keeping a single novnc window / tab open.
