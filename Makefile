.ENV := .venv
.VERSION := 1.0
.TAG := $(VERSION).$(shell date +'%Y%m%d%H%M')
.DEFAULT_GOAL := help
.PHONY: help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	pytest -s

pylint:
	pylint --rcfile=.pylintrc src

coverage: ## Run tests with coverage
	pytest --cov=src --cov-report=term-missing