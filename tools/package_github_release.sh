#!/bin/zsh
set -eu

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
repo=${0:A:h:h}
cd "$repo"

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
stamp=$(date +%Y%m%d_%H%M%S)
stage=$(mktemp -d /tmp/ucp_pse_release.XXXXXX)
archive="$repo/UCP-PSE-source-${stamp}.tar.gz"
manifest="$repo/UCP-PSE-source-${stamp}.sha256"

# 段落说明：定义可复用的脚本函数，封装这一阶段的编译、运行或汇总操作。
cleanup() { rm -rf "$stage"; }
trap cleanup EXIT

# 段落说明：创建本次发布或实验需要的输出目录；已有目录保持不变。
mkdir -p "$stage/UCP-PSE"
mkdir -p "$stage/UCP-PSE/CED_schedule" "$stage/UCP-PSE/standalone_ced" \
  "$stage/UCP-PSE/tools"

# Publication whitelist: keep executable source and current documentation;
# exclude exploratory notes, raw data/results, binaries, IDE state, old
# snapshots, and third-party trees whose provenance is described in the docs.
cp README.md EXPERIMENT_RUNBOOK.md .clang-format "$stage/UCP-PSE/"
cp -R docs "$stage/UCP-PSE/"
cp CED_schedule/Config.h CED_schedule/Multimethod.h \
  CED_schedule/Population.h CED_schedule/Problems.h \
  CED_schedule/RecentSchedulingAlgorithms.h CED_schedule/main.cpp \
  CED_schedule/Multimethod.cpp CED_schedule/MultimethodAde.cpp \
  CED_schedule/MultimethodMeme.cpp \
  CED_schedule/Problems.cpp CED_schedule/main_recent_algorithms.cpp \
  CED_schedule/RecentSchedulingAlgorithms.cpp \
  "$stage/UCP-PSE/CED_schedule/"
cp standalone_ced/README.md standalone_ced/REPRODUCTION_AUDIT.md \
  standalone_ced/benchmark_config.h standalone_ced/ced_problem.h \
  standalone_ced/mpi_support.h standalone_ced/ced_problem.cpp \
  standalone_ced/madde.cpp standalone_ced/qphh.cpp \
  standalone_ced/fca_g.cpp standalone_ced/soea.cpp \
  standalone_ced/hga.cpp standalone_ced/bipop_cde.cpp \
  standalone_ced/amtsa.cpp "$stage/UCP-PSE/standalone_ced/"
cp tools/main_industrial_benchmark_compact.cpp \
  tools/run_component_vs_nine_metrics_10x.sh \
  tools/summarize_component_vs_nine_metrics_10x.sh \
  tools/package_github_release.sh "$stage/UCP-PSE/tools/"

# 段落说明：生成发布归档并计算 SHA-256，便于下载后验证内容未被改变。
tar -C "$stage" -czf "$archive" UCP-PSE
shasum -a 256 "$archive" > "$manifest"
print -r -- "$archive"
print -r -- "$manifest"
