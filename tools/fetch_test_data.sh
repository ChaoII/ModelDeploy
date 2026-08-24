#!/usr/bin/env bash
# 一键下载 ModelDeploy 测试数据(test_data.zip)并解压到仓库根 test_data/
set -euo pipefail

URL="${URL:-https://www.modelscope.cn/models/ChaoII0987/ModelDeploy_cmake_deps/resolve/master/test_data.zip}"
FORCE=0
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  OUTDIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
TEST_DATA_DIR="$OUTDIR/test_data"
ZIP="$OUTDIR/test_data.zip"

if [[ -d "$TEST_DATA_DIR" && "$FORCE" -eq 0 ]]; then
  echo "test_data/ 已存在(用 --force 重新下载)。"
  exit 0
fi

echo "下载 $URL -> $ZIP"
if ! curl -L -o "$ZIP" "$URL"; then
  echo "下载失败" >&2
  rm -f "$ZIP"
  exit 1
fi

echo "解压 $ZIP -> $OUTDIR"
if ! unzip -o -q "$ZIP" -d "$OUTDIR"; then
  echo "解压失败(需要 unzip)" >&2
  rm -f "$ZIP"
  exit 1
fi
rm -f "$ZIP"
echo "完成: $TEST_DATA_DIR"
