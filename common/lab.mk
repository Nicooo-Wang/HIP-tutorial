# 各 Lab 的 Makefile 只需写：include ../common/lab.mk
COMMON := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LAB    := $(abspath .)
A      ?=
CASE   ?=

.PHONY: grade run verify clean
grade:
	@python3 $(COMMON)/grade.py $(LAB) $(if $(A),--only $(A))

# make run A=a2 [CASE=name]：编译并直接运行（人类可读输出，便于调试/打印）。
# 没设置任何 GPU 可见性变量时，和 make grade 一样自动选最空闲的卡（grade.py --pick-gpu）。
run:
	@test -n "$(A)" || (echo "用法: make run A=a2 [CASE=用例名]"; exit 1)
	@d=$$(ls -d $(LAB)/$(A)-*/ 2>/dev/null | head -n 1); \
	if [ -z "$$d" ]; then echo "错误: 找不到 Assignment $(A)（$(LAB) 下没有 $(A)-* 目录）"; exit 2; fi; \
	n=$$(ls $$d | grep -c '\.hip$$'); \
	if [ "$$n" -ne 1 ]; then echo "错误: $$(basename $$d) 中应恰好有一个 .hip 文件，实际 $$n 个；请把多余的文件改名为别的后缀（如 .hip.bak）或移出该目录"; exit 2; fi; \
	src=$$(ls $$d*.hip); mkdir -p $(LAB)/build; \
	/opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942 -I$(COMMON) -DSTUDENT_SRC="\"$$src\"" \
	  $$d/test.cpp -o $(LAB)/build/run-$(A) || exit 1; \
	g=$$(python3 $(COMMON)/grade.py --pick-gpu); \
	if [ -n "$$g" ]; then export HIP_VISIBLE_DEVICES=$$g; fi; \
	$(LAB)/build/run-$(A) $(CASE)

verify:
	@python3 $(COMMON)/verify.py $(LAB)

clean:
	rm -rf $(LAB)/build
