from beyondGD.neural import descent, evolve, swarm, simplex

from beyondGD.utils import time_track, dict_max
from beyondGD.tasks.utils import (
    setup,
    init_population,
    population_from_model,
    evaluate,
)

# --- map tasks to string args
tasks: dict = {
    "descent": descent,
    "evolve": evolve,
    "swarm": swarm,
    "simplex": simplex,
}

#
#
#  -------- do_train -----------
#
@time_track
def do_train(args: dict) -> None:

    # --- setup experiment
    model, data, utils = setup(args)

    # create empty population, return type holder
    population: dict = {}
    last_return_type: str = None

    # load model from file
    if utils["data_config"].get("load_model"):
        try:
            model = utils.get("model_class").load(
                utils["data_config"].get("load_model")
            )
            last_return_type: str = "model"

        except FileNotFoundError:
            print(
                "\n[--- File not found, continuing with fresh model. ---]"
            )

    # log that training is orchestra
    if len(utils.get("train_config").get("tasks")) > 1:
        print("\n[--- ORCHESTRA ---]")

    # --- start training
    for task in utils.get("train_config").get("tasks"):

        # --- init population, if is first task and not gradient descent
        if (
            task.get("type") not in ("descent", "swarm")
            and not last_return_type
        ):
            population = init_population(
                utils.get("model_class"),
                utils.get("model_config"),
                task.get("population_size"),
            )

        # --- create population from last task model
        elif (
            task.get("type") != "descent"
            and last_return_type == "model"
        ):
            population = population_from_model(
                utils.get("model_class"),
                model,
                task.get("population_size"),
            )

        # --- start task
        print(f"\n[--- {task.get('type').upper()} ---]")

        # handle task, which take and return population
        if task.get("type") in ("evolve", "simplex"):
            population = tasks.get(task.get("type"))(
                population,
                data.get("train"),
                data.get("dev"),
                **task.get("parameters"),
            )

            last_return_type = "population"

        # handle task, which take and return model
        elif task.get("type") in ("descent", "swarm"):

            # if last task has returned a population, extract the best model
            if last_return_type == "population":
                best, _ = dict_max(population)

            # else use the model as best
            else:
                best = model

            # train gradient descent
            model = tasks.get(task.get("type"))(
                best,
                data.get("train"),
                data.get("dev"),
                **task.get("parameters"),
            )

            last_return_type = "model"

    # --- get best model from population
    if last_return_type == "population":
        best, _ = dict_max(population)

    # --- last model equals best model
    else:
        best = model

    # --- run metric
    evaluate(
        best,
        utils.get("encoding"),
        data.get("test"),
    )

    # --- save model
    if utils["data_config"].get("save_model"):
        best.save(utils["data_config"].get("save_model"))
