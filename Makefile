module := beyondGD

# config files:
model_config := config_example/model.json
data_config := config_example/data.json

descent_config := config_example/train_descent.json
evolve_config := config_example/train_evolution.json
swarm_config := config_example/train_swarm.json
simplex_config := config_example/train_simplex.json
orchestra_config := config_example/train_orchestra.json


# data server:
data_server := https://simon-muenker.de

descent:
	@python3 -m ${module} -M ${model_config} -T ${descent_config} -D ${data_config}

evolve:
	@python3 -m ${module} -M ${model_config} -T ${evolve_config} -D ${data_config}

swarm:
	@python3 -m ${module} -M ${model_config} -T ${swarm_config} -D ${data_config}

simplex:
	@python3 -m ${module} -M ${model_config} -T ${simplex_config} -D ${data_config}

orchestra:
	@python3 -m ${module} -M ${model_config} -T ${orchestra_config} -D ${data_config}

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
	@wget  ${data_server}/data/en_proof_of_concept.conllu -P ./data

lint:
	@pre-commit run --all-files

clean:
	rm -rf logs cache .pytest_cache .mypy_cache
	clear
