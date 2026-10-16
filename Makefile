LABS := lab0-env lab1-hip-basics lab2-cktile-intro lab3-cktile-concepts
.PHONY: verify-all test-infra
test-infra:
	python3 common/test_grade.py
verify-all: test-infra
	@set -e; for l in $(LABS); do echo "######## $$l"; $(MAKE) -s -C $$l verify; done
