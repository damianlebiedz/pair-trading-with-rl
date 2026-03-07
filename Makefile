.PHONY: full_track fast_track fetch backtest train backtest_rl download_artifacts

ifndef RUN_MODE
$(error Environment not specified! Please use: 'make full_track RUN_MODE=docker' or 'make fast_track RUN_MODE=docker')
endif

ifneq ($(RUN_MODE),docker)
ifneq ($(RUN_MODE),poetry)
$(error Unknown RUN_MODE=$(RUN_MODE). Available modes are: docker, poetry)
endif
endif

ZENODO_URL = "https://zenodo.org/record/XXXXXXX/files/reproduction_package.zip"

ifeq ($(RUN_MODE), poetry)
	CMD_FETCH = poetry run python helpers/generate_assets_list.py && poetry run python helpers/fetch_historical_data.py
	CMD_BACKTEST = poetry run python runners/run_backtest.py
	CMD_TRAIN = poetry run python runners/train_agent.py
	CMD_BACKTEST_RL = poetry run python runners/run_backtest.py use_rl=True
else
	CMD_FETCH = docker compose run --rm generate_assets_list && docker compose run --rm fetch_historical_data
	CMD_BACKTEST = docker compose run --rm run_backtest
	CMD_TRAIN = docker compose run --rm train_agent
	CMD_BACKTEST_RL = docker compose run --rm run_backtest python runners/run_backtest.py use_rl=True
endif

full_track: fetch backtest train backtest_rl
	@echo "=== Full reproduction track successfully completed ==="

fast_track: download_artifacts backtest backtest_rl
	@echo "=== Fast track evaluation successfully completed ==="

download_artifacts:
	@echo "=== Stage 0: Downloading frozen dataset and pre-trained model ==="
	@mkdir -p data results/models
	curl -L -o reproduction_package.zip $(ZENODO_URL)
	unzip -o reproduction_package.zip -d .
	rm reproduction_package.zip
	@echo "=== Artifacts successfully downloaded and extracted ==="

fetch:
	@echo "=== Stage 1: Data fetching (Binance API) ==="
	$(CMD_FETCH)

backtest:
	@echo "=== Stage 2: Base Backtest (without RL) ==="
	$(CMD_BACKTEST)

train:
	@echo "=== Stage 3: RL Training ==="
	$(CMD_TRAIN)

backtest_rl:
	@echo "=== Stage 4: Backtest (with RL model) ==="
	$(CMD_BACKTEST_RL)