## Installation and test

```
git clone git@github.com:psFournier/dl_toolbox.git
cd dl_toolbox
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -m ipykernel install --user --name=dl_toolbox
```

Then launch in a jupyterlab the notebook train_semcity.ipynb in dl_toolbox/train with the dl_toolbox kernel, it should run from scratch.