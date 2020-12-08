module := geneticNLP

# config files:
tagger_config := config_example/postagger.json
train_config := config_example/train_descent.json
evolve_config := config_example/train_evolution.json
swarm_config := config_example/train_swarm.json
amoeba_config := config_example/train_amoeba.json
data_config := config_example/data.json

# data server:
data_server := https://simon-muenker.de

descent:
	@python3 -m ${module} descent -M ${tagger_config} -T ${train_config} -D ${data_config}

evolve:
	@python3 -m ${module} evolve -M ${tagger_config} -T ${evolve_config} -D ${data_config}

swarm:
	@python3 -m ${module} swarm -M ${tagger_config} -T ${swarm_config} -D ${data_config}

amoeba:
	@python3 -m ${module} amoeba -M ${tagger_config} -T ${amoeba_config} -D ${data_config}

test:
	@python3 -m pytest -s -v

install:
	@pip3 install -r requirements.txt

download:
	@mkdir -p ./data
	@wget  ${data_server}/data/cc.en.32.bin -P ./data
	@wget  ${data_server}/data/en_partut-ud-train.conllu -P ./data
	@wget  ${data_server}/data/en_partut-ud-dev.conllu -P ./data
	@wget  ${data_server}/data/en_partut-ud-test.conllu -P ./data

lint:
	@pre-commit run --all-files

clean:
	rm -rf logs cache .pytest_cache .mypy_cache
	clear
