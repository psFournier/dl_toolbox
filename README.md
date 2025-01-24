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
jupyter-lab --no-browser --ip=0.0.0.0
```

Now in a browser go to the URL indicated by your console.
Then open src/segmenter.ipynb and select the kernel dl_toolbox in the top right corner. 
The notebook should run.