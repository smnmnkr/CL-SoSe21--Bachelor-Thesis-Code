module := geneticNLP

# config files:
tagger_config := config_example/postagger.json
evolve_config := config_example/evolution.json
train_config := config_example/training.json
data_config := config_example/data.json

evolve:
	@python3 -m ${module} evolve -M ${tagger_config} -E ${evolve_config} -D ${data_config}

train:
	@python3 -m ${module} train -M ${tagger_config} -T ${train_config} -D ${data_config}

hybrid:
	@python3 -m ${module} hybrid -M ${tagger_config} -T ${train_config} -E ${evolve_config} -D ${data_config}

test:
	@python3 -m pytest -s -v

install:
	@pip3 install -r requirements.txt

lint:
	@pre-commit run --all-files

clean:
	rm -rf logs cache .pytest_cache .mypy_cache
	clear
