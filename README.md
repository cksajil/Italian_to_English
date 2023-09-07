# Italian_to_English
## Text translation from Italian to English 

<img src="https://i.ibb.co/pzBzP2M/attention-concat.png">

**Python Version**
```
Python 3.9.1
```

### Setting up virtual environment

```console
# Installing Virtual Environment
python -m pip install --user virtualenv

# Creating New Virtual Environment
python -m venv envname

# Activating Virtual Environment
source envname/bin/activate

# Upgrade PIP
python -m pip install --upgrade pip

# Installing Packages
python -m pip install -r requirements.txt
```

### How to run

```console
# Example input
python main.py --it "Com'è il tempo?"

# Expected output
What's the weather like?
```