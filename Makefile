SHELL=/bin/bash
PROJECT_NAME=gem
PROJECT_PATH=gem/ tests/ eval/ examples/
LINT_PATHS=${PROJECT_PATH}

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

lint:
	$(call check_install, isort)
	$(call check_install, pylint)
	isort --check --diff --project=${LINT_PATHS}
	pylint -j 8 --recursive=y ${LINT_PATHS}

format:
	$(call check_install, autoflake)
	autoflake --remove-all-unused-imports -i -r ${LINT_PATHS}
	$(call check_install, black)
	black ${LINT_PATHS}
	$(call check_install, isort)
	isort ${LINT_PATHS}
