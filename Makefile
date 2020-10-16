
evolve:
	@python3 -m geneticNLP evolve -M config_example/postagger.json -E config_example/evolution.json -D config_example/data.json

train:
	@python3 -m geneticNLP train -M config_example/postagger.json -T config_example/training.json -D config_example/data.json

test:
	@python3 -m pytest -s -v

install:
	@pip3 install -r requirements.txt

lint:
	@pre-commit run --all-files

clean:
	rm -rf logs cache .pytest_cache .mypy_cache
	clear
