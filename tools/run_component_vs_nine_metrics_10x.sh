#!/bin/zsh
set -eu

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
repo=${0:A:h:h}
cd "$repo"

# 段落说明：创建本次发布或实验需要的输出目录；已有目录保持不变。
mpicxx="$repo/.local/mpich/bin/mpic++"
mpirun="$repo/.local/mpich/bin/mpirun"
outdir="$repo/results/component_vs_nine_metrics_10x_20260816"
bindir="$repo/build/component_vs_nine_metrics_10x_20260816"
mkdir -p "$outdir" "$bindir"

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
scales=(1000 10000 100000 1000000)
methods=(component_policy madde qphh fca_g soea_bbrl hga bipop_cde amtsa nlshade_lbc slpso_ars)
seeds=(20260616 20260617 20260618 20260619 20260620 20260621 20260622 20260623 20260624 20260625)

# 段落说明：定义可复用的脚本函数，封装这一阶段的编译、运行或汇总操作。
compile_component() {
  local scale=$1
  local exe="$bindir/component_policy_t${scale}"
  [[ -x "$exe" ]] && return
  "$mpicxx" -std=c++17 -O2 -DUSE_MPI -DTNUM=$scale -DMAXGEN=50 \
    -DPOPSIZE=8 -DVERBOSE_OUTPUT=0 -DPAPER_COMPONENT_POLICY=1 \
    CED_schedule/main.cpp CED_schedule/Multimethod.cpp \
    CED_schedule/MultimethodMeme.cpp CED_schedule/Problems.cpp -o "$exe"
}

# 段落说明：定义可复用的脚本函数，封装这一阶段的编译、运行或汇总操作。
compile_recent() {
  local scale=$1
  local exe="$bindir/recent_t${scale}"
  [[ -x "$exe" ]] && return
  "$mpicxx" -std=c++17 -O2 -DUSE_MPI -DTNUM=$scale -DMAXGEN=50 \
    -DPOPSIZE=8 -DVERBOSE_OUTPUT=0 -DFIXED_GENERATION_BUDGET=1 \
    CED_schedule/main_recent_algorithms.cpp CED_schedule/Multimethod.cpp \
    CED_schedule/MultimethodMeme.cpp CED_schedule/RecentSchedulingAlgorithms.cpp \
    CED_schedule/Problems.cpp -o "$exe"
}

# 段落说明：定义可复用的脚本函数，封装这一阶段的编译、运行或汇总操作。
compile_standalone() {
  local method=$1
  local source=$2
  local scale=$3
  local exe="$bindir/${method}_t${scale}"
  [[ -x "$exe" ]] && return
  "$mpicxx" -std=c++17 -O2 -DUSE_MPI -DTNUM=$scale -DMAXGEN=50 \
    "standalone_ced/${source}.cpp" standalone_ced/ced_problem.cpp \
    CED_schedule/Problems.cpp -o "$exe"
}

# 段落说明：遍历论文规定的规模、方法或配对随机种子，逐项执行同一流程。
for scale in $scales; do
  compile_component $scale
  compile_recent $scale
  compile_standalone madde madde $scale
  compile_standalone qphh qphh $scale
  compile_standalone fca_g fca_g $scale
  compile_standalone soea_bbrl soea $scale
  compile_standalone hga hga $scale
  compile_standalone bipop_cde bipop_cde $scale
  compile_standalone amtsa amtsa $scale
done

# 段落说明：遍历论文规定的规模、方法或配对随机种子，逐项执行同一流程。
for scale in $scales; do
  for method in $methods; do
    # 控制说明：处理这一算法或动作分支，完成后退出当前分派。
    case "$method" in
      component_policy) exe="$bindir/component_policy_t${scale}" ;;
      nlshade_lbc|slpso_ars) exe="$bindir/recent_t${scale}" ;;
      *) exe="$bindir/${method}_t${scale}" ;;
    esac
    for run in {1..10}; do
      logfile="$outdir/${method}_t${scale}_run${run}.txt"
      if [[ -s "$logfile" ]] && rg -q '^Industrial device load:' "$logfile"; then
        print -r -- "SKIP method=$method scale=$scale run=$run"
        continue
      fi
      print -r -- "RUN method=$method scale=$scale run=$run"
      tmpfile="$logfile.tmp"
      rm -f "$tmpfile"
      args=(env -u CED_GLOBAL_SEARCH_ENABLED -u CED_GLOBAL_SEARCH_ALG
            -u CED_MEME_SEARCH_MODE CED_SEED=${seeds[$run]})
      if [[ "$method" == nlshade_lbc || "$method" == slpso_ars ]]; then
        args+=(CED_COMPARISON_ALG=$method)
      fi
      $args "$mpirun" -np 8 "$exe" > "$tmpfile"
      rg -q '^The best solution = ' "$tmpfile"
      rg -q '^Raw energy = ' "$tmpfile"
      rg -q '^Industrial device load:' "$tmpfile"
      mv "$tmpfile" "$logfile"
      rg '^The best solution = |^Raw energy = |^Makespan = |^Time = ' "$logfile"
    done
  done
done

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
"$repo/tools/summarize_component_vs_nine_metrics_10x.sh"
