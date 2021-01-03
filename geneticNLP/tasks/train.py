from geneticNLP.neural import descent, evolve, swarm, amoeba

from geneticNLP.utils import time_track, dict_max
from geneticNLP.tasks.utils import (
    setup,
    init_population,
    evaluate,
)

# --- map tasks to string args
tasks: dict = {
    "descent": descent,
    "evolve": evolve,
    "swarm": swarm,
    "amoeba": amoeba,
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

    # log that training is orchestra
    if len(utils.get("train_config").get("tasks")) > 1:
        print("\n[--- ORCHESTRA ---]")

    # --- start training
    for task in utils.get("train_config").get("tasks"):

        # --- init population, if is first task and not gradient descent
        if task.get("type") != ("descent") and not population:
            population = init_population(
                utils.get("model_class"),
                utils.get("model_config"),
                task.get("population_size"),
            )

        # --- create population from last task model
        if last_return_type == "model":
            # TODO: get population from model
            pass

        # --- start task
        print(f"\n[--- {task.get('type').upper()} ---]")

        # handle task, which take and return population
        if task.get("type") == ("evolve" or "amoeba"):
            population = tasks.get(task.get("type"))(
                population,
                data.get("train"),
                data.get("dev"),
                **task.get("parameters"),
            )

            last_return_type = "population"

        # handle task, which take population and return model
        elif task.get("type") == "swarm":
            model = tasks.get(task.get("type"))(
                population,
                data.get("train"),
                data.get("dev"),
                **task.get("parameters"),
            )

            last_return_type = "model"

        # handle task, which take and return model
        elif task.get("type") == "descent":

            if last_return_type != None:
                best, _ = dict_max(population)

            else:
                best = model

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
