.PHONY: pipeline fetch backtest train backtest_rl

ifndef RUN_MODE
$(error Environment not specified! Please use: 'make pipeline RUN_MODE=docker' or 'make pipeline RUN_MODE=poetry')
endif

ifneq ($(RUN_MODE),docker)
ifneq ($(RUN_MODE),poetry)
$(error Unknown RUN_MODE=$(RUN_MODE). Available modes are: docker, poetry)
endif
endif

ifeq ($(RUN_MODE), poetry)
	# --- LOCAL COMMANDS (POETRY) ---
	CMD_FETCH = poetry run python helpers/generate_assets_list.py && poetry run python helpers/fetch_historical_data.py
	CMD_BACKTEST = poetry run python runners/run_backtest.py
	CMD_TRAIN = poetry run python runners/train_agent.py
	CMD_BACKTEST_RL = poetry run python runners/run_backtest.py performance.use_rl=true
else
	# --- DOCKER COMPOSE COMMANDS ---
	CMD_FETCH = docker compose run --rm fetch_historical_data
	CMD_BACKTEST = docker compose run --rm run_backtest
	CMD_TRAIN = docker compose run --rm train_agent
	CMD_BACKTEST_RL = docker compose run --rm run_backtest python runners/run_backtest.py performance.use_rl=true
endif

pipeline: fetch backtest train backtest_rl
	@echo "=== Full workflow successfully completed ==="

fetch:
	@echo "=== Stage 1: Data fetching ==="
	$(CMD_FETCH)

backtest:
	@echo "=== Stage 2: Backtest (without RL) ==="
	$(CMD_BACKTEST)

train:
	@echo "=== Stage 3: RL Training ==="
	$(CMD_TRAIN)

backtest_rl:
	@echo "=== Stage 4: Backtest (with RL) ==="
	$(CMD_BACKTEST_RL)