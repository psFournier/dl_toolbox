## Installation and test

Cloner le dépôt puis:

```
cd dl_toolbox
git checkout -b branche_perso
git push -u origin branche_perso
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -m ipykernel install --user --name=dl_toolbox
```

Then launch in a jupyterlab the notebook train_semcity.ipynb in dl_toolbox/train with the dl_toolbox kernel, it should run from scratch.