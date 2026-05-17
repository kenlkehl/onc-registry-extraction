from __future__ import annotations

from onc_registry_pipeline.config import PipelineConfig
from onc_registry_pipeline.convert import load_name_map
from onc_registry_pipeline.dictionary.loader import NAACCRDictionary


def test_pipeline_config_defaults_find_vendored_reference_data_from_other_cwd(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = PipelineConfig()

    assert config.data_items_csv.exists()
    assert config.code_list_csv.exists()
    assert config.alternate_names_csv.exists()
    assert (config.seer_manuals_dir / "manifest.json").exists()

    dictionary = NAACCRDictionary(config)
    dictionary.load()

    date_of_birth = dictionary.get_item(240)
    assert date_of_birth is not None
    assert date_of_birth.name == "Date of Birth"


def test_convert_readable_names_find_vendored_dictionary_from_other_cwd(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    name_map = load_name_map()

    assert name_map["dateOfBirth"] == "Date of Birth"
