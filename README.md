# Colorization
## Development

### 1. Clone Repository
```bash
$ git clone https://github.com/alayshahh/Colorization.git
$ cd Colorization
```

### 2. Setup Virtual Environment
- If you don't have `venv` installed, run the following
  ```bash
  $ python3 -m pip install --user virtualenv
  ```

```bash
$ python3 -m venv colorization_venv
```

### 3. Activate Virtual Environment
```bash
$ source colorization_venv/bin/activate
```

### 4. Install packages
```
(probabilistic_hunting_venv) $ pip install -r requirements.txt
```

### 5. Update `requirements.txt`
_Note: Only do this whenever you add a new lib or package to the project so that the change is reflected on github_

```bash
(probabilistic_hunting_venv) $ pip freeze > requirements.txt
```
