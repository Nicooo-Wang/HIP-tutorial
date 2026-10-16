# 各 Lab 的 Makefile 只需写：include ../common/lab.mk
COMMON := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LAB    := $(abspath .)
A      ?=
CASE   ?=

.PHONY: grade run verify clean
grade:
	@python3 $(COMMON)/grade.py $(LAB) $(if $(A),--only $(A))

# make run A=a2 [CASE=name]：编译并直接运行（人类可读输出，便于调试/打印）
run:
	@test -n "$(A)" || (echo "用法: make run A=a2 [CASE=用例名]"; exit 1)
	@d=$$(ls -d $(LAB)/$(A)-*/); src=$$(ls $$d*.hip); mkdir -p $(LAB)/build; \
	/opt/rocm/bin/hipcc -std=c++20 -O2 --offload-arch=gfx942 -I$(COMMON) -DSTUDENT_SRC="\"$$src\"" \
	  $$d/test.cpp -o $(LAB)/build/run-$(A) && $(LAB)/build/run-$(A) $(CASE)

verify:
	@python3 $(COMMON)/verify.py $(LAB)

clean:
	rm -rf $(LAB)/build
