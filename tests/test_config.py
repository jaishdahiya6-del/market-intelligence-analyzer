from src import config


def test_set_keys_and_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    (tmp_path / ".env").write_text("OTHER=1\nCOINGECKO_API_KEY=old\n")
    config.set_keys(alpha="AV1", coingecko="CG1", cg_type="demo", persist=True)
    txt = (tmp_path / ".env").read_text()
    assert "OTHER=1" in txt and "COINGECKO_API_KEY=CG1" in txt and "ALPHA_VANTAGE_API_KEY=AV1" in txt
    assert config.alpha_key() == "AV1" and config.coingecko_key() == "CG1"
