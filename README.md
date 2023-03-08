create venv and install reqs

```sh
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

either download model using `download_model.sh` or put model at this directory.

then run the demo

```
GRADIO_SERVER_PORT=8888 python3 demo.py
```
