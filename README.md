## DIS WEB APP for background remove

<br>

## This is app use dis v1.0 dataset model.the official DIS project repo is link below:

[**Project Page**](https://xuebinqin.github.io/dis/index.html)

<br>

# Required

Install Docker Desktop<br>

# Install

## 1.Clone this repo

use git

## 2.build container

go to the DIS_WEB directory, and do command.
then, installed all requrements.

```
docker-compose build
```

# Run app

## 1. Go into container

```
docker-compose up -d
```

```
docker-compose exec dis-web /bin/bash
```

## 2. Run APP

```
python app.py
```

Dealt server setting is 127.0.0.1:7860 <br>
If you want to use other setting, please edit app.py.<br>
server_name=url server_port=port

```python
gr.Interface(
    fn=inference,
    inputs=gr.Image(type='filepath'),
    outputs=["image", "image"],
    examples=[['dax.jpg'], ['goya.jpg']],
    title=title,
    description=description,
    article=article,
    allow_flagging='never',
    theme="default",
    cache_examples=False,
    ).launch(server_name="0.0.0.0", server_port=7860, enable_queue=True, debug=True)
```

<br>

Access and enjoy use.
