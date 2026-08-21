#!/bin/zsh
set -eu

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
repo=${0:A:h:h}
outdir="$repo/results/component_vs_nine_metrics_10x_20260816"
raw="$outdir/raw_metrics.csv"
summary="$outdir/summary_metrics.csv"

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
print 'method,scale,run,objective,energy,makespan,time_s,cloud_mean,cloud_sd,cloud_min,cloud_median,cloud_p95,cloud_max,cloud_jain,edge_mean,edge_sd,edge_min,edge_median,edge_p95,edge_max,edge_jain,device_mean,device_sd,device_min,device_median,device_p95,device_max,device_jain' > "$raw"

# 段落说明：遍历论文规定的规模、方法或配对随机种子，逐项执行同一流程。
for file in "$outdir"/*_t*_run*.txt(N); do
  name=${file:t:r}
  method=${name%%_t<->*}
  suffix=${name#${method}_t}
  scale=${suffix%%_run*}
  run=${suffix##*_run}
  awk -v method="$method" -v scale="$scale" -v run="$run" '
  function value_after_equal(line, a) { split(line,a,"= "); split(a[2],a," "); return a[1] }
  function loads(line, out, a,b,i) {
    sub(/^.*load: /,"",line); split(line,a,", ");
    # 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (i=1;i<=7;i++) { split(a[i],b,"="); out[i]=b[2] }
  }
  /^The best solution = / {objective=value_after_equal($0)}
  /^Raw energy = / {energy=value_after_equal($0)}
  /^Makespan = / {makespan=value_after_equal($0)}
  /^Time = / {time=value_after_equal($0)}
  /^Cloud load:/ {loads($0,c)}
  /^Edge load:/ {loads($0,e)}
  /^Industrial device load:/ {loads($0,d)}
  END {
    # 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (objective!="" && d[7]!="") {
      printf "%s,%s,%s,%s,%s,%s,%s",method,scale,run,objective,energy,makespan,time;
      # 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for(i=1;i<=7;i++) printf ",%s",c[i];
      # 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for(i=1;i<=7;i++) printf ",%s",e[i];
      # 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for(i=1;i<=7;i++) printf ",%s",d[i];
      printf "\n";
    }
  }' "$file" >> "$raw"
done

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
awk -F, '
BEGIN {
  OFS=",";
  print "method,scale,n,objective_mean,objective_sd,objective_best,energy_mean,makespan_mean,time_mean_s,time_sd_s,cloud_mean,cloud_sd,cloud_jain,edge_mean,edge_sd,edge_jain,device_mean,device_sd,device_jain"
}
NR>1 {
  key=$1 SUBSEP $2; n[key]++;
  # 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for(i=4;i<=28;i++){s[key,i]+=$i; ss[key,i]+=$i*$i}
  # 控制说明：依据目标值决定接受、最优更新或审计路径。
  if(!(key in best) || $4<best[key]) best[key]=$4
}
END {
  # 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for(key in n){
    split(key,a,SUBSEP); no=n[key];
    osd=no>1?sqrt((ss[key,4]-s[key,4]*s[key,4]/no)/(no-1)):0;
    tsd=no>1?sqrt((ss[key,7]-s[key,7]*s[key,7]/no)/(no-1)):0;
    print a[1],a[2],no,s[key,4]/no,osd,best[key],s[key,5]/no,s[key,6]/no,s[key,7]/no,tsd,s[key,8]/no,s[key,9]/no,s[key,14]/no,s[key,15]/no,s[key,16]/no,s[key,21]/no,s[key,22]/no,s[key,23]/no,s[key,28]/no;
  }
}' "$raw" | { IFS= read -r header; print -r -- "$header"; sort -t, -k2,2n -k1,1; } > "$summary"

# 段落说明：执行本阶段的路径设置、参数准备或文件处理。
print -r -- "$summary"
