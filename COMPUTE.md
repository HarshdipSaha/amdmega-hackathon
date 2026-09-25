I have access to do(i have confirmed on my amd developer portal, so we can use this):


Infrastructure
Getting access to a GPU
Important: You do not need your own AMD hardware, AMD Developer Cloud credits, or Fireworks API credits to take part in a challenge.
GPU Infrastructure can be accessed at https://notebooks.amd.com/hackathon
You will need to sign-in via AMD Dev Program SSO credentials and the first launch can be done pressing "Launch Notebook" button on the page. Launching one gives you a JupyterLab session on a GPU node: you start the notebook, wait for it to become ready, and open it. The first redirected URL when you open a session is single-use and short-lived, so open your session from a fresh tab and visiting https://notebooks.amd.com/hackathon each time rather than bookmarking the URL.


Time quota
Sessions are capped at 3 hours of running time each day. Every 24 hours the quota refreshes. A session left idle still consumes that quota, so press the “Turn-off Session” button when you stop working. The quota is time the pod exists, not time you spend typing.


Storage
Each user gets around 25gb of persistent storage. The location of persistent storage depends on the pod url (very important):
- If your redirected URL name has “jupyter-hack-***” in it then your persistent storage will be at /persistent and only files/folders up to about 25gb stored in /persistent location of your jupyter-lab environment survives quota exhaustion and turn-off session. Files stored under /workspace will not be persisted for those users/teams.
- If your redirected URL has “rgapi-hackathon-***” in it then our persistent storage will be at /workspace and only files/folders up to about 25gb stored in /workspace location of your jupyter-lab environment survives quota exhaustion and turn-off session. Files stored under any other location will not be persisted for those users/teams.