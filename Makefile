LABS := lab0-env lab1-hip-basics lab2-cktile-intro lab3-cktile-concepts
.PHONY: verify-all test-infra compdb
# compdb：生成 compile_commands.json，供 VSCode 的 clangd 做补全/跳转（见 .vscode/settings.json）。
compdb:
	python3 common/gen_compile_commands.py
# test-infra：判分器单元测试 + harness 自检（common/harness_selftest.cpp）。
# 自检里 always_fails_for_protocol_check 是故意失败的用例（检查结果协议），其余用例必须全部 OK。
test-infra:
	python3 common/test_grade.py
	@mkdir -p build
	/opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942 -Icommon common/harness_selftest.cpp -o build/harness_selftest
	@g=$$(python3 common/grade.py --pick-gpu); if [ -n "$$g" ]; then export HIP_VISIBLE_DEVICES=$$g; fi; \
	./build/harness_selftest > build/harness_selftest.log 2>&1; cat build/harness_selftest.log; \
	bad=$$(grep -E '^[^ ].* FAIL' build/harness_selftest.log | awk '{print $$1}' | grep -vx 'always_fails_for_protocol_check'); \
	if [ -n "$$bad" ] || ! grep -Eq '^always_fails_for_protocol_check +FAIL' build/harness_selftest.log \
	   || ! grep -q '^通过 ' build/harness_selftest.log; then \
	  echo "harness 自检失败：$${bad:-自检程序没有正常跑完}"; exit 1; fi; \
	echo "harness 自检 OK（always_fails_for_protocol_check 按预期失败，其余全部 OK）"
verify-all: test-infra
	@set -e; for l in $(LABS); do echo "######## $$l"; $(MAKE) -s -C $$l verify; done
