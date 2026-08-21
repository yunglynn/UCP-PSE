#!/bin/zsh
# 下载四种论文数据资产，校验压缩包，然后解压并再次校验原始矩阵。
set -eu

# 以脚本所在仓库为根目录，避免调用者当前目录影响输出位置。
repo=${0:A:h:h}
data_dir="$repo/data"
release_url="https://github.com/yunglynn/UCP-PSE/releases/download/datasets-v1"
assets=(datamatrix_1000.gz datamatrix_10000.gz datamatrix_100000.gz
        datamatrix_1000000_compact.gz)

# 保留下载文件，便于断点后重新校验；curl -C - 从已有字节继续下载。
mkdir -p "$data_dir"
for asset in $assets; do
  print -r -- "Downloading $asset"
  curl -fL -C - "$release_url/$asset" -o "$data_dir/$asset"
done

# 首先检查网络下载的压缩资产是否与发布者清单完全一致。
cd "$data_dir"
shasum -a 256 -c ASSET_SHA256SUMS

# 解压到求解器默认的相对路径；gzip 文件继续保留，便于重复部署。
for asset in $assets; do
  output=${asset%.gz}
  print -r -- "Extracting $asset -> $output"
  gzip -dc "$asset" > "$output.tmp"
  mv "$output.tmp" "$output"
done

# 再检查未压缩矩阵，确保解压、磁盘写入和换行均未改变数据。
shasum -a 256 -c SHA256SUMS
print -r -- "All four UCP-PSE datasets are ready in $data_dir"
