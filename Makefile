# Thin wrappers around commands that are already written down in README.md, AGENTS.md and
# .github/workflows/terminal-minimal-server.yml. Nothing here is new. Run `make` or `make help`.
# The targets that need the GPU, docker or a downloaded model were not run when this file was written
# (no GPU in that session): they are marked "not run without GPU".

PY     ?= python3
CONFIG ?= strata-unsloth-ud-iq1_s.json
H      := $(notdir $(CURDIR))

.DEFAULT_GOAL := help
.PHONY: help image build setup serve test test-serve smoke deps-dev intelinference

help: ## Show this list
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

image: ## Build the strata-sycl-dev image, about 6 min once, 13 GB (README "Install" step 2; not run without GPU)
	docker build --pull -t strata-sycl-dev sycl/tools

build: ## Build the engine: AOT for bmg-g31, ends with BUILD EXIT 0 (README "Install" step 2; not run without GPU)
	docker run --rm -u $$(id -u):$$(id -g) -v "$(CURDIR)/..:/work" -e AOT=bmg-g31 -e REPO=/work/$(H) \
	    -e BUILD_DIR=/work/$(H)/build-sycl-aot strata-sycl-dev "cd /work/$(H) && bash sycl/tools/build.sh strata"

setup: ## Download the served model, write the config and start it (README "Install" step 3; not run without GPU)
	./setup.sh --backend sycl --family unsloth --model UD-IQ1_S --context 262144 --vision no

serve: ## Start the server on a config: make serve CONFIG=strata-<model>.json (AGENTS.md "Serve"; not run without GPU)
	$(PY) serve/server.py --engine strata --config $(CONFIG)

test: ## Terminal tests, standard library unittest, no GPU
	PYTHONDONTWRITEBYTECODE=1 $(PY) -m unittest discover -s tests/terminal -t tests/terminal

test-serve: ## Server tests as CI runs them (needs jinja2 and regex: make deps-dev), no GPU
	PYTHONDONTWRITEBYTECODE=1 $(PY) -m unittest discover -s serve -p 'test_*.py'

smoke: ## Check that the command line starts
	$(PY) strata.py --help
	$(PY) strata.py launch --help

deps-dev: ## Install the runtime and dev requirements into the Python that PY names
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

intelinference: ## Build the terminal front end (Rust, needs cargo) to intelinference/target/release/intelinference
	cargo build --release --manifest-path intelinference/Cargo.toml
