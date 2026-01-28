#!/bin/bash
#
# PP Test Suite Runner
# 
# Runs comprehensive PP benchmarks to measure:
# 1. Per-rank recv blocking time
# 2. Decode-phase ITL and throughput
# 3. Comparison between baseline and optimized
#
# Usage:
#   ./04_run_all.sh --mode baseline      # Run without optimization
#   ./04_run_all.sh --mode optimized     # Run with SGLANG_PP_NONBLOCKING_RECV=1
#   ./04_run_all.sh --mode async_depth   # Run with --pp-async-batch-depth 2
#   ./04_run_all.sh --mode compare       # Run all and compare
#

set -e

# Configuration
MODEL="Qwen/Qwen3-0.6B"
PORT=30000
PP_SIZE=2
CHUNKED_PREFILL_SIZE=256
NUM_REQUESTS=20
OUTPUT_LEN=128
WARMUP=3

# Directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SGLANG_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SGLANG_DIR/pp_test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 --mode <baseline|optimized|async_depth|compare>"
    echo ""
    echo "Modes:"
    echo "  baseline    - Run without any PP optimization"
    echo "  optimized   - Run with SGLANG_PP_NONBLOCKING_RECV=1"
    echo "  async_depth - Run with --pp-async-batch-depth 2"
    echo "  compare     - Run all modes and compare results"
    exit 1
}

# Parse arguments
MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --num-requests)
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --output-len)
            OUTPUT_LEN="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$MODE" ]; then
    usage
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

kill_server() {
    echo -e "${YELLOW}Killing any existing sglang servers...${NC}"
    pkill -9 -f sglang 2>/dev/null || true
    sleep 3
}

wait_for_server() {
    local url="$1"
    local max_wait=120
    local waited=0
    
    echo "Waiting for server at $url..."
    while [ $waited -lt $max_wait ]; do
        if curl -s "$url/health" > /dev/null 2>&1; then
            echo -e "${GREEN}Server is ready!${NC}"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo "  Waiting... ($waited/$max_wait seconds)"
    done
    
    echo -e "${RED}Server failed to start within $max_wait seconds${NC}"
    return 1
}

run_benchmark() {
    local config_name="$1"
    local extra_env="$2"
    local extra_args="$3"
    
    local results_subdir="$RESULTS_DIR/${config_name}_$TIMESTAMP"
    mkdir -p "$results_subdir"
    
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}Running: $config_name${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo "Results dir: $results_subdir"
    
    # Kill existing server
    kill_server
    
    # Start server with profiling enabled
    local log_file="$results_subdir/server.log"
    echo "Starting server..."
    echo "  Model: $MODEL"
    echo "  PP Size: $PP_SIZE"
    echo "  Extra env: $extra_env"
    echo "  Extra args: $extra_args"
    
    cd "$SGLANG_DIR"
    source .venv/bin/activate
    
    (
        eval "$extra_env" python -m sglang.launch_server \
            --model-path "$MODEL" \
            --port "$PORT" \
            --pipeline-parallel-size "$PP_SIZE" \
            --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
            $extra_args \
            2>&1 | tee "$log_file"
    ) &
    
    local server_pid=$!
    
    # Wait for server to be ready
    if ! wait_for_server "http://localhost:$PORT"; then
        echo -e "${RED}Server failed to start. Check $log_file${NC}"
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    
    # Small delay to ensure server is fully ready
    sleep 5
    
    # Run decode micro-benchmark
    echo -e "\n${YELLOW}Running decode micro-benchmark...${NC}"
    python "$SCRIPT_DIR/02_decode_microbench.py" \
        --url "http://localhost:$PORT" \
        --mode sequential \
        --num-requests "$NUM_REQUESTS" \
        --output-len "$OUTPUT_LEN" \
        --warmup "$WARMUP" \
        --export "$results_subdir/decode_results.json" \
        2>&1 | tee "$results_subdir/decode_benchmark.log"
    
    # Run recv timing analysis
    echo -e "\n${YELLOW}Running recv timing analysis...${NC}"
    
    # Generate more requests for timing data
    python "$SCRIPT_DIR/01_baseline_recv_timing.py" \
        --url "http://localhost:$PORT" \
        --num-requests 30 \
        --output-len 64 \
        2>&1 | tee "$results_subdir/workload.log"
    
    # Analyze the server log for timing data
    python "$SCRIPT_DIR/01_baseline_recv_timing.py" \
        --analyze "$log_file" \
        --export "$results_subdir/recv_timing.json" \
        2>&1 | tee "$results_subdir/recv_analysis.log"
    
    # Stop server
    echo "Stopping server..."
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    kill_server
    
    echo -e "${GREEN}Results saved to: $results_subdir${NC}"
}

run_baseline() {
    run_benchmark "baseline" "" ""
}

run_optimized() {
    run_benchmark "nonblocking" "SGLANG_PP_NONBLOCKING_RECV=1 SGLANG_PP_DEBUG=1" ""
}

run_async_depth() {
    run_benchmark "async_depth_2" "SGLANG_PP_DEBUG=1" "--pp-async-batch-depth 2"
}

compare_results() {
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}COMPARING RESULTS${NC}"
    echo -e "${GREEN}============================================${NC}"
    
    # Find latest results for each config
    local baseline_dir=$(ls -td "$RESULTS_DIR"/baseline_* 2>/dev/null | head -1)
    local optimized_dir=$(ls -td "$RESULTS_DIR"/nonblocking_* 2>/dev/null | head -1)
    local async_dir=$(ls -td "$RESULTS_DIR"/async_depth_2_* 2>/dev/null | head -1)
    
    echo ""
    echo "Latest results:"
    [ -n "$baseline_dir" ] && echo "  Baseline: $baseline_dir"
    [ -n "$optimized_dir" ] && echo "  Nonblocking: $optimized_dir"
    [ -n "$async_dir" ] && echo "  Async depth: $async_dir"
    
    # Compare decode results if available
    if [ -n "$baseline_dir" ] && [ -f "$baseline_dir/decode_results.json" ]; then
        echo ""
        echo "=== BASELINE DECODE ==="
        python3 -c "
import json
with open('$baseline_dir/decode_results.json') as f:
    d = json.load(f)
print(f'Throughput: {d[\"throughput_tps\"]:.1f} tok/s')
print(f'Mean ITL: {d[\"mean_itl_ms\"]:.2f} ms')
print(f'Median ITL: {d[\"median_itl_ms\"]:.2f} ms')
print(f'P95 ITL: {d[\"p95_itl_ms\"]:.2f} ms')
"
    fi
    
    if [ -n "$optimized_dir" ] && [ -f "$optimized_dir/decode_results.json" ]; then
        echo ""
        echo "=== NONBLOCKING RECV DECODE ==="
        python3 -c "
import json
with open('$optimized_dir/decode_results.json') as f:
    d = json.load(f)
print(f'Throughput: {d[\"throughput_tps\"]:.1f} tok/s')
print(f'Mean ITL: {d[\"mean_itl_ms\"]:.2f} ms')
print(f'Median ITL: {d[\"median_itl_ms\"]:.2f} ms')
print(f'P95 ITL: {d[\"p95_itl_ms\"]:.2f} ms')
"
    fi
    
    if [ -n "$async_dir" ] && [ -f "$async_dir/decode_results.json" ]; then
        echo ""
        echo "=== ASYNC DEPTH 2 DECODE ==="
        python3 -c "
import json
with open('$async_dir/decode_results.json') as f:
    d = json.load(f)
print(f'Throughput: {d[\"throughput_tps\"]:.1f} tok/s')
print(f'Mean ITL: {d[\"mean_itl_ms\"]:.2f} ms')
print(f'Median ITL: {d[\"median_itl_ms\"]:.2f} ms')
print(f'P95 ITL: {d[\"p95_itl_ms\"]:.2f} ms')
"
    fi
    
    # Compare recv timing if available
    if [ -n "$baseline_dir" ] && [ -f "$baseline_dir/recv_timing.json" ]; then
        echo ""
        echo "=== BASELINE RECV TIMING ==="
        python3 -c "
import json
with open('$baseline_dir/recv_timing.json') as f:
    d = json.load(f)
for k, v in d.items():
    print(f'{k}: mean={v[\"mean_us\"]:.0f}us, p95={v[\"p95_us\"]:.0f}us')
" 2>/dev/null || echo "  (no timing data)"
    fi
    
    if [ -n "$async_dir" ] && [ -f "$async_dir/recv_timing.json" ]; then
        echo ""
        echo "=== ASYNC DEPTH 2 RECV TIMING ==="
        python3 -c "
import json
with open('$async_dir/recv_timing.json') as f:
    d = json.load(f)
for k, v in d.items():
    print(f'{k}: mean={v[\"mean_us\"]:.0f}us, p95={v[\"p95_us\"]:.0f}us')
" 2>/dev/null || echo "  (no timing data)"
    fi
}

# Main
case $MODE in
    baseline)
        run_baseline
        ;;
    optimized)
        run_optimized
        ;;
    async_depth)
        run_async_depth
        ;;
    compare)
        run_baseline
        run_async_depth
        run_optimized
        compare_results
        ;;
    results)
        compare_results
        ;;
    *)
        usage
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"
