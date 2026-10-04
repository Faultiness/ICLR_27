"""SHIR engine construction for the VitalDB and WeatherBench2 tasks."""

from dataclasses import replace

from .adapters.vitaldb import VitalDBAdapter, make_vitaldb_task
from .adapters.weather import RegionalMeanPredictor, WeatherAdapter, make_weather_task
from .cards import ReferenceCardAdapter
from .engine import EngineConfig, SHIREngine
from .llm import LLMProposalBackend


def make_engine(dataset, predictor, *, predictor_id, operator_reference,
                card_reference, llm_config):
    """Create SHIR with a supplied frozen predictor.

    VitalDB: standardized values + observed masks [30, 40] -> MAP [3] in mmHg.
    WeatherBench2: standardized [12, 9, 41, 61] -> fields [3, 21, 31] in K.
    Adapters normalize inputs; the predictor must not normalize them again.
    """
    if dataset == "vitaldb":
        base_adapter = VitalDBAdapter(operator_reference)
        task = make_vitaldb_task(predictor_id, base_adapter)
        model = lambda payload: predictor(base_adapter.model_input(payload))
        budgets = (1, 2, 2, 3)
    elif dataset == "weatherbench2":
        base_adapter = WeatherAdapter(operator_reference)
        task = make_weather_task(predictor_id, base_adapter)
        model = RegionalMeanPredictor(lambda payload: predictor(base_adapter.model_input(payload)))
        budgets = (5, 7, 9, 14)
    else:
        raise ValueError("dataset must be vitaldb or weatherbench2")
    adapter = ReferenceCardAdapter(base_adapter, card_reference, adapter_id=task.adapter_id)
    task = replace(task, adapter_id=f"{task.adapter_id}:card:{card_reference.fingerprint}")
    return SHIREngine(task, model, adapter, LLMProposalBackend(llm_config),
                      EngineConfig(explanation_budgets=budgets))
