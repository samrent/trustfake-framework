USER := $(shell whoami)
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

export USER
export USER_ID
export GROUP_ID
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.DEFAULT_GOAL := help

#############################
# Help and environment info #
#############################

.PHONY: help
help: ## Show this help message
	@echo -e "$(GREEN)TrustFake Project Makefile$(NC)"
	@echo -e "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

env-info: ## Display environment information
	@echo -e "$(GREEN)Environment Information:$(NC)"
	@echo -e "  User: $(USER)"
	@echo -e "  User ID: $(USER_ID)"
	@echo -e "  Group ID: $(GROUP_ID)"

###################
# Docker commands #
###################

.PHONY: build
build: ## Build Docker containers
	@echo -e "$(GREEN)Building Docker container...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
		docker compose build

.PHONY: rebuild
rebuild: ## Rebuild Docker containers
	@echo -e "$(GREEN)Rebuilding Docker container...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
		docker compose build --no-cache

.PHONY: logs
logs: ## Follow all container logs
	@echo -e "$(GREEN)Showing all container logs...$(NC)"
	docker compose logs -f

.PHONY: up
up: ## Launch trustfake-dev container
	@echo -e "$(GREEN)Launching container from trustfake-dev image...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker compose up -d trustfake-dev

.PHONY: shell
shell: ## Launch shell in dev container
	@echo -e "$(GREEN)Launching zsh shell in dev container...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker exec -it -u ${USER_ID}:${GROUP_ID} ${USER}_trustfake_dev zsh

.PHONY: down
down: ## Stop trustfake-dev container
	@echo -e "$(GREEN)Stop docker dev container...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker compose down --remove-orphans trustfake-dev

.PHONY: jupyter
jupyter: ## Launch a Jupyter Lab server in the dev container (http://localhost:8800)
	@echo -e "$(GREEN)Starting Jupyter Lab in dev container on http://localhost:8800 ...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker compose up -d trustfake-dev
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker compose exec -u ${USER_ID}:${GROUP_ID} trustfake-dev \
		jupyter lab --ip=0.0.0.0 --port=8800 --no-browser --ServerApp.token=''

################
# Code quality #
################

.PHONY: dev-ruff-check
dev-ruff-check: ## Run ruff check in trustfake container
	@echo -e "$(GREEN)Running ruff check in trustfake container...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker compose run --rm trustfake-dev uvx ruff check /home/workspace/

.PHONY: dev-ruff-fix
dev-ruff-fix: ## Run ruff check with auto-fix in trustfake container
	@echo -e "$(GREEN)Running ruff check with auto-fix in trustfake container...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker compose run --rm trustfake-dev uvx ruff check /home/workspace/ --fix

.PHONY: dev-ruff-format
dev-ruff-format: ## Run ruff format in trustfake container
	@echo -e "$(GREEN)Running ruff format in trustfake container...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker compose run --rm trustfake-dev uvx ruff format /home/workspace/

#########
# Tests #
#########

.PHONY: dev-test-attacks
dev-test-attacks: ## Run the adversarial attack contract tests in trustfake container
	@echo -e "$(GREEN)Running attack contract tests in trustfake container...$(NC)"
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) USER=$(USER) \
	docker compose exec -u ${USER_ID}:${GROUP_ID} trustfake-dev pytest tests/attacks -v