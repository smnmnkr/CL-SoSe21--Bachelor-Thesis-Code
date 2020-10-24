from geneticNLP.neural import hybrid
from geneticNLP.utils import load_json, time_track
from geneticNLP.tasks.utils import setup

from geneticNLP.models.postagger import POSstripped


#
#
#  -------- do_hybrid -----------
#
@time_track
def do_hybrid(args: dict) -> None:

    # --- load config json files
    model_config: dict = load_json(args.model_config)
    data_config: dict = load_json(args.data_config)

    # --- setup experiment
    model, data = setup(model_config, data_config)

    # --- start hybrid
    hybrid(
        model,
        data.get("train"),
        data.get("dev"),
    )
