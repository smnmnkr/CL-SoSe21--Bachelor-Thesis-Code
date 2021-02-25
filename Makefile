module := beyondGD

#
# config files:
model_config := config_example/model.json
data_config := config_example/data.json

descent_config := config_example/train_descent.json
evolve_config := config_example/train_evolution.json
swarm_config := config_example/train_swarm.json
simplex_config := config_example/train_simplex.json
orchestra_config := config_example/train_orchestra.json

#
# data server:
data_server := https://simon-muenker.de

#
# demos:
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

download:
	@mkdir -p ./data
	@wget  ${data_server}/data/cc.en.32.bin -P ./data
	@wget  ${data_server}/data/en_partut-ud-train.conllu -P ./data
	@wget  ${data_server}/data/en_partut-ud-dev.conllu -P ./data
	@wget  ${data_server}/data/en_partut-ud-test.conllu -P ./data
	@wget  ${data_server}/data/en_proof_of_concept.conllu -P ./data


#
# util:
test:
	@python3 -m pytest -s -v

install:
	@pip3 install -r requirements.txt

lint:
	@pre-commit run --all-files

clean:
	rm -rf logs cache .pytest_cache .mypy_cache
	clear


#
# experiments:
# ToDo: find more elegant solution
exp0_path := results/00-Baseline-Gradient/config
exp1_path := results/01-Baseline-Evolve/config
exp2_path := results/02-Baseline-Swarm/config
exp3_path := results/03-Baseline-Simplex/config
exp4_path := results/04-Orchestration/config

exp0:
	@python3 -m ${module} -M ${exp0_path}/model.json -T ${exp0_path}/train.json -D ${exp0_path}/data.json

exp1:
	@python3 -m ${module} -M ${exp1_path}/model.json -T ${exp1_path}/train.json -D ${exp1_path}/data.json

exp2:
	@python3 -m ${module} -M ${exp2_path}/model.json -T ${exp2_path}/train.json -D ${exp2_path}/data.json

exp3:
	@python3 -m ${module} -M ${exp3_path}/model.json -T ${exp3_path}/train.json -D ${exp3_path}/data.json

exp4:
	@python3 -m ${module} -M ${exp4_path}/model.json -T ${exp4_path}/train_evolve.json -D ${exp4_path}/data.json
	@python3 -m ${module} -M ${exp4_path}/model.json -T ${exp4_path}/train_swarm.json -D ${exp4_path}/data.json
	@python3 -m ${module} -M ${exp4_path}/model.json -T ${exp4_path}/train_simplex.json -D ${exp4_path}/data.json