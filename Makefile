.PHONY: app

format: black flake

env:
	python3 -m venv env

install:
	python -m pip install -r requirements.txt

install-dev: install
	python -m pip install -r requirements-dev.txt

black:
	black app/main.py

flake:
	flake8 --ignore=E501 app/main.py

app:
	streamlit run app/main.py
