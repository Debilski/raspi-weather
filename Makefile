env      ?= localhost,
playbook ?= ansible/setup.yaml

install:
	ansible-playbook -i $(env) $(playbook)

dry-run: ## make dry-run playbook=setup # Run a playbook in dry run mode
	ansible-playbook -i $(env) --diff --check $(playbook)

